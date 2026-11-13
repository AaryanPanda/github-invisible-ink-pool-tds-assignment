"""Invisible Ink predictor (IITM TDS bonus).

  python invisible_ink.py train pool.jsonl [more_labelled.jsonl ...] --model ink_model.pkl
  python invisible_ink.py predict test/*.json --model ink_model.pkl --out out

Pipeline
  1. Pen-down ("ink") detector: two-stage gradient boosting on per-sample motion features
     (speed, turning, straightness, timing, undo/clear events).
  2. Letter reader: HOG features + SVC on letter images (synthetic template letters + real letters,
     real ones weighted 3x), trained on out-of-fold ink so it sees realistic, imperfect ink.
  3. Decoder: dynamic programming over left-to-right letter cuts (length 3-5, or the given length).
Outputs drawing_N.png (predicted ink + auto guess) and guesses.txt
"""
import json, os, io, glob, gzip, pickle, argparse, itertools
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.feature import hog
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

ALPHA = 'ACEFHJKMNPRTUVWXY'; JUNK = len(ALPHA)
FEAT_VERSION = 2
SEED = 0
rng = np.random.default_rng(SEED)

# ---------------- labels: which pen-down samples survive undo/clear ----------------
def runs(p):
    d=np.diff(np.r_[0,p.astype(int),0]); return list(zip(np.where(d==1)[0],np.where(d==-1)[0]))
def kept_runs(l):
    """simulate undo/clear: return list of (a,b) sample-index runs that survive"""
    s=np.array(l['samples']); rs=runs(s[:,3]>0.5)
    ev=[(s[a,0],'start',(a,b)) for a,b in rs]+[(c['t'],c['type'],None) for c in l['controls'] if c['type'] in('undo','clear')]
    ev.sort(key=lambda e:e[0]); stack=[]
    for t,ty,r in ev:
        if ty=='start': stack.append(r)
        elif ty=='undo' and stack: stack.pop()
        elif ty=='clear': stack=[]
    return stack


# ---------------- per-sample motion features ----------------
def target(l):
    s=np.array(l['samples']); y=np.zeros(len(s))
    for a,b in kept_runs(l): y[a:b]=1
    return y
def roll(a,w,f=np.mean):
    a=np.asarray(a,float); n=len(a)
    p=np.pad(a,(w,w),mode='constant',constant_values=np.nan)
    win=np.lib.stride_tricks.sliding_window_view(p,2*w+1)
    g={np.mean:np.nanmean,np.std:np.nanstd,np.max:np.nanmax,np.min:np.nanmin}.get(f,f)
    return g(win,axis=1)
def feats(l,fv=2):
    s=np.array(l['samples'],float); t=s[:,0]; x=s[:,2]; y=s[:,1]; n=len(s)
    # normalise position by robust bbox
    x0,x1=np.percentile(x,[2,98]); y0,y1=np.percentile(y,[2,98]); sc=max(x1-x0,y1-y0,1)
    xn=(x-x0)/sc; yn=(y-y0)/sc
    dt=np.r_[50,np.diff(t)]; dx=np.r_[0,np.diff(x)]; dy=np.r_[0,np.diff(y)]
    v=np.hypot(dx,dy)/np.maximum(dt,1)*1000
    ang=np.arctan2(dy,dx); turn=np.abs(np.angle(np.exp(1j*np.r_[0,np.diff(ang)])))
    F={'xn':xn,'yn':yn,'tf':(t-t[0])/(t[-1]-t[0]+1),'dt':dt,'dx':dx/sc,'dy':dy/sc,'v':v/sc,'turn':turn,'vabs':v}
    for w in (2,4,8,16):
        F[f'vm{w}']=roll(v,w)/sc; F[f'vs{w}']=roll(v,w,np.std)/sc; F[f'tm{w}']=roll(turn,w)
        F[f'dtm{w}']=roll(dt,w)
        # straightness: displacement / path length over window
        lo=np.clip(np.arange(n)-w,0,n-1); hi=np.clip(np.arange(n)+w,0,n-1)
        cum=np.r_[0,np.cumsum(np.hypot(np.diff(x),np.diff(y)))]
        F[f'st{w}']=np.hypot(x[hi]-x[lo],y[hi]-y[lo])/(cum[hi]-cum[lo]+1e-6)
        F[f'ddx{w}']=(x[hi]-x[lo])/sc; F[f'ddy{w}']=(y[hi]-y[lo])/sc
    for k in (-3,-2,-1,1,2,3):
        F[f'v_l{k}']=np.roll(v,k)/sc; F[f'dt_l{k}']=np.roll(dt,k)
    # controls
    und=np.array(sorted(c['t'] for c in l['controls'] if c['type'] in('undo','clear')))
    if len(und):
        nxt=np.array([und[und>=ti].min()-ti if (und>=ti).any() else 1e6 for ti in t])
        prv=np.array([ti-und[und<=ti].max() if (und<=ti).any() else 1e6 for ti in t])
        after=np.array([(und>ti).sum() for ti in t])
    else: nxt=prv=np.full(n,1e6); after=np.zeros(n)
    F['tnext_undo']=np.minimum(nxt,6e4); F['tprev_undo']=np.minimum(prv,6e4); F['n_undo_after']=after
    F['n_undo_total']=np.full(n,len(und)); F['dur']=np.full(n,t[-1]-t[0])
    if fv==1: F["length"]=np.full(n,float(l.get('length') or 4))   # old models only; carried no signal
    names=list(F); return np.c_[[F[k] for k in names]].T.astype(float), names


# ---------------- letter images ----------------
R=24
def word_frame(l,w):
    s=np.array(l['samples'],float); x=s[:,2]; y=s[:,1]
    m=w>0.4
    if m.sum()<5: m=w>=np.percentile(w,70)
    y0,y1=np.percentile(y[m],[3,97]); x0,x1=np.percentile(x[m],[2,98])
    return x,y,(x0,x1,y0,y1)
def raster(x,y,w,xa,xb,y0,y1):
    img=np.zeros((R,R)); h=max(y1-y0,1e-3); wd=max(xb-xa,1e-3)
    sc=max(h,wd); ox=(sc-wd)/2; oy=(sc-h)/2
    inn=(x>=xa)&(x<=xb)
    ww=np.minimum(w[:-1],w[1:]); keep=inn[:-1]&inn[1:]&(ww>=0.05)
    if keep.any():
        i=np.where(keep)[0]; f=np.linspace(0,1,6)[None,:]
        px=((x[i,None]+(x[i+1,None]-x[i,None])*f)-xa+ox)/sc*(R-1)
        py=((y[i,None]+(y[i+1,None]-y[i,None])*f)-y0+oy)/sc*(R-1)
        wv=np.repeat(ww[i],6)
        px=px.ravel(); py=py.ravel(); ok=(px>=0)&(px<R)&(py>=0)&(py<R)
        np.add.at(img,(py[ok].astype(int),px[ok].astype(int)),wv[ok])
    img=gaussian_filter(img,1.0); mx=img.max()
    return img/mx if mx>0 else img
def seg_feat(x,y,w,xa,xb,y0,y1,X0,X1):
    img=raster(x,y,w,xa,xb,y0,y1)
    inn=(x>=xa)&(x<=xb); mass=(w*inn).sum()/(w.sum()+1e-6)
    return np.r_[img.ravel(),(xb-xa)/max(y1-y0,1e-3),mass,(xb-xa)/max(X1-X0,1e-3)]
def true_cuts(l):
    """x cut positions between letters using kept strokes partitioned in time into `length` groups"""
    s=np.array(l['samples'],float); x=s[:,2]; k=l['length']; rs=kept_runs(l)
    rs=sorted(rs)
    if len(rs)<k: return None
    ext=[(np.percentile(x[a:b],5),np.percentile(x[a:b],95)) for a,b in rs]
    n=len(rs); import itertools
    best=None
    for cut in itertools.combinations(range(1,n),k-1):
        g=[list(range(a,b)) for a,b in zip((0,)+cut,cut+(n,))]
        spans=[(min(ext[j][0] for j in gg),max(ext[j][1] for j in gg)) for gg in g]
        cost=sum(b-a for a,b in spans)
        # penalise order violation
        cost+=sum(max(0,spans[i][1]-spans[i+1][0])*3 for i in range(k-1))
        if best is None or cost<best[0]: best=(cost,spans)
    sp=best[1]
    return [(sp[i][1]+sp[i+1][0])/2 for i in range(k-1)], sp


def letter_examples(l,w,jit=3):
    r=true_cuts(l)
    if r is None: return [],[]
    cuts,sp=r; x,y,(X0,X1,y0,y1)=word_frame(l,w)
    edges=[X0-0.02*(X1-X0)]+cuts+[X1+0.02*(X1-X0)]
    F=[];Y=[]; H=y1-y0
    for i,ch in enumerate(l['code']):
        a,b=edges[i],edges[i+1]
        for j in range(jit):
            d=0 if j==0 else 0.08
            aa=a+rng.normal(0,d)*H; bb=b+rng.normal(0,d)*H
            if bb>aa: F.append(seg_feat(x,y,w,aa,bb,y0,y1,X0,X1)); Y.append(ALPHA.index(ch))
        # junk: half letters and merged pairs
        mid=(a+b)/2
        for aa,bb in [(a,mid),(mid,b)]:
            F.append(seg_feat(x,y,w,aa,bb,y0,y1,X0,X1)); Y.append(JUNK)
        if i+1<len(l['code']):
            F.append(seg_feat(x,y,w,a,edges[i+2],y0,y1,X0,X1)); Y.append(JUNK)
            F.append(seg_feat(x,y,w,mid,(edges[i+1]+edges[i+2])/2,y0,y1,X0,X1)); Y.append(JUNK)
    return F,Y
def pos_score(g,t):
    return sum(a==b for a,b in zip(g,t))/len(t)

# ---------------- synthetic template letters ----------------
def arc(cx,cy,rx,ry,a0,a1,n=12):
    a=np.linspace(a0,a1,n); return list(zip(cx+rx*np.cos(a),cy+ry*np.sin(a)))
P_bowl=[(0,0),(0.6,0)]+arc(0.6,0.23,0.35,0.23,-np.pi/2,np.pi/2,8)+[(0,0.46)]
T={
'A':[[[(0,1),(0.5,0),(1,1)],[(0.22,0.6),(0.78,0.6)]]],
'C':[[arc(0.55,0.5,0.5,0.5,-0.25*np.pi,-1.75*np.pi,16)]],
'E':[[[(0.9,0),(0,0),(0,1),(0.9,1)],[(0,0.5),(0.7,0.5)]],[[(0,0),(0,1)],[(0,0),(0.9,0)],[(0,0.5),(0.7,0.5)],[(0,1),(0.9,1)]]],
'F':[[[(0.9,0),(0,0),(0,1)],[(0,0.5),(0.7,0.5)]],[[(0,0),(0,1)],[(0,0),(0.9,0)],[(0,0.5),(0.7,0.5)]]],
'H':[[[(0,0),(0,1)],[(1,0),(1,1)],[(0,0.5),(1,0.5)]]],
'J':[[[(0.85,0),(0.85,0.7)]+arc(0.5,0.7,0.35,0.3,0,np.pi,8)],[[(0.2,0),(1,0)],[(0.65,0),(0.65,0.7)]+arc(0.35,0.7,0.3,0.3,0,np.pi,8)]],
'K':[[[(0,0),(0,1)],[(0.9,0),(0,0.55),(0.9,1)]],[[(0,0),(0,1)],[(0.9,0),(0,0.55)],[(0.25,0.42),(0.9,1)]]],
'M':[[[(0,1),(0,0),(0.5,0.6),(1,0),(1,1)]],[[(0,1),(0.15,0),(0.5,0.7),(0.85,0),(1,1)]]],
'N':[[[(0,1),(0,0),(1,1),(1,0)]],[[(0,0),(0,1)],[(0,0),(1,1),(1,0)]]],
'P':[[[(0,1),(0,0)],P_bowl],[[(0,1)]+P_bowl]],
'R':[[[(0,1),(0,0)],P_bowl+[(0.9,1)]],[[(0,1)]+P_bowl,[(0.3,0.46),(0.9,1)]]],
'T':[[[(0,0),(1,0)],[(0.5,0),(0.5,1)]]],
'U':[[[(0,0),(0,0.6)]+arc(0.5,0.6,0.5,0.4,np.pi,0,10)+[(1,0)]],
     [[(0,0),(0,0.75)]+arc(0.5,0.75,0.5,0.25,np.pi,0,8)+[(1,0)]]],
'V':[[[(0,0),(0.5,1),(1,0)]]],
'W':[[[(0,0),(0.25,1),(0.5,0.3),(0.75,1),(1,0)]]],
'X':[[[(0,0),(1,1)],[(1,0),(0,1)]]],
'Y':[[[(0,0),(0.5,0.5),(1,0)],[(0.5,0.5),(0.5,1)]],[[(0,0),(0.5,0.5)],[(1,0),(0.5,0.5),(0.5,1)]],[[(0,0),(0.5,0.5)],[(1,0),(0,1)]]],
}
rng2=np.random.default_rng(1)
def synth(ch):
    strokes=T[ch][rng2.integers(len(T[ch]))]
    aw=rng2.uniform(0.45,1.0); slant=rng2.normal(0,0.15); rot=rng2.normal(0,0.08)
    pts=[];W=[]
    prev=None
    for st in strokes:
        st=np.array(st,float); st[:,0]*=aw
        # densify
        seg=[]
        for a,b in zip(st[:-1],st[1:]):
            n=max(2,int(np.hypot(*(b-a))*18*rng2.uniform(0.6,1.4)))
            seg+=list(a+(b-a)*f for f in np.linspace(0,1,n,endpoint=False))
        seg.append(st[-1]); seg=np.array(seg)
        seg+=np.cumsum(rng2.normal(0,0.012,seg.shape),0)*0.5+rng2.normal(0,0.012,seg.shape)
        if prev is not None:   # pen-up travel
            n=rng2.integers(2,8); tr=np.array([prev+(seg[0]-prev)*f for f in np.linspace(0,1,n+2)[1:-1]])
            tr+=rng2.normal(0,0.05,tr.shape); pts+=list(tr); W+=list(rng2.uniform(0,0.45,len(tr)))
        pts+=list(seg); W+=list(np.clip(rng2.normal(0.85,0.2,len(seg)),0.1,1)); prev=seg[-1]
    p=np.array(pts); W=np.array(W)
    # drop-outs in ink
    k=rng2.random(len(W))<0.08; W[k]*=0.2
    x=p[:,0]+slant*(p[:,1]-0.5); y=p[:,1]
    c,s=np.cos(rot),np.sin(rot); x,y=c*x-s*y,s*x+c*y
    y0,y1=y.min(),y.max(); x0,x1=x.min(),x.max(); H=y1-y0
    xa=x0-rng2.normal(0.05,0.06)*H; xb=x1+rng2.normal(0.05,0.06)*H
    ya=y0+rng2.normal(0,0.05)*H; yb=y1+rng2.normal(0,0.05)*H
    return seg_feat(x,y,W,xa,xb,ya,yb,0,4*H)
def hogf(F):
    F=np.atleast_2d(np.array(F)); imgs=F[:,:R*R].reshape(-1,R,R)
    H=np.array([hog(im,orientations=9,pixels_per_cell=(6,6),cells_per_block=(2,2)) for im in imgs])
    return np.c_[H,F[:,R*R:R*R+2]]
def synth_set(n=150):
    F=[];Y=[]
    for i,ch in enumerate(ALPHA):
        for _ in range(n): F.append(synth(ch)); Y.append(i)
    return np.array(F),np.array(Y)


# ---------------- input handling ----------------
def norm_rec(l):
    """Make any recording look like a training file. Remembers whether the file had controls."""
    l=dict(l)
    l['_has_controls']=isinstance(l.get('controls'),list)
    l['samples']=[(list(r)+[0]*(4-len(r)))[:4] for r in l['samples']]
    for r in l['samples']:
        if r[3] is None: r[3]=0
    if not isinstance(l.get('controls'),list): l['controls']=[]
    l.setdefault('length',None)
    return l
def parse_text(txt,name=''):
    txt=txt.strip().lstrip('\ufeff')
    try:
        o=json.loads(txt); objs=o if isinstance(o,list) else [o]
    except json.JSONDecodeError:
        objs=[json.loads(x) for x in txt.splitlines() if x.strip()]
    out=[]
    for o in objs:
        if isinstance(o,dict) and isinstance(o.get('samples'),list) and len(o['samples'])>=5:
            o['_file']=name; out.append(norm_rec(o))
    return out
def read_any(paths):
    out=[]
    for p in paths:
        out+=parse_text(open(p,encoding='utf-8').read(),os.path.basename(p))
    return out
def dedup(L):
    seen=set(); out=[]
    for l in L:
        key=(l.get('code'),tuple(round(v,1) for r in l['samples'] for v in r[1:3]))
        if key not in seen: seen.add(key); out.append(l)
    return out

# ---------------- training ----------------
def stackf(p):
    return np.c_[[roll(p,w) for w in (1,3,6,12)]+[np.roll(p,k) for k in (-4,-2,2,4)]+[roll(p,6,np.max),roll(p,6,np.min)]].T
def hgb(n=300): return HistGradientBoostingClassifier(max_iter=n,learning_rate=0.05,l2_regularization=1.0,random_state=SEED)
def train_ink(L, fv=FEAT_VERSION, K=4):
    """Two-stage ink detector. Returns (m1, m2, oof): oof = out-of-fold stage-1 ink per recording,
    used to train the letter reader (softer ink than stage 2; tested to give better letter accuracy)."""
    Xs=[feats(l,fv)[0] for l in L]; Ys=[target(l) for l in L]; idx=np.arange(len(L)); fold=idx%K
    oof=[None]*len(L)
    for g in range(K):
        m=hgb().fit(np.vstack([Xs[i] for i in idx[fold!=g]]),np.concatenate([Ys[i] for i in idx[fold!=g]]))
        for i in idx[fold==g]: oof[i]=m.predict_proba(Xs[i])[:,1]
        del m
    m1=hgb().fit(np.vstack(Xs),np.concatenate(Ys))
    m2=hgb(200).fit(np.vstack([np.c_[Xs[i],stackf(oof[i])] for i in idx]),np.concatenate(Ys))
    return m1,m2,oof
def train(paths, out):
    L=dedup([l for l in read_any(paths) if l.get('code')])
    print(f'training on {len(L)} unique labelled recordings')
    m1,m2,oof=train_ink(L)
    print('ink model done; building letter classifier (synthetic + real)...')
    SF,SY=synth_set(200); Xl=[hogf(SF)]; Yl=[SY]; Wl=[np.ones(len(SY))]
    for l,w in zip(L,oof):
        for ww in (w,target(l)):
            ff,yy=letter_examples(l,ww,jit=2)
            if not ff: continue
            ff=np.array(ff);yy=np.array(yy);m=yy!=JUNK
            Xl.append(hogf(ff[m])); Yl.append(yy[m]); Wl.append(np.full(m.sum(),3.0))   # real letters count 3x
    clf=make_pipeline(StandardScaler(),SVC(C=2,gamma='scale',probability=True,random_state=SEED)).fit(
        np.vstack(Xl),np.concatenate(Yl),svc__sample_weight=np.concatenate(Wl))
    M={'v':FEAT_VERSION,'m1':m1,'m2':m2,'clf':clf,'n':len(L)}
    with gzip.open(out,'wb',compresslevel=6) as f: pickle.dump(M,f,protocol=5)
    print('saved',out)

# ---------------- prediction ----------------
class SoftSVC:
    """Kept only so model files made by an earlier experimental version still load."""
    def __init__(self,sc,svc): self.sc=sc; self.svc=svc
    def predict_proba(self,X):
        d=self.svc.decision_function(self.sc.transform(np.asarray(X,np.float32)))
        full=np.full((len(d),JUNK+1),-1e3); full[:,self.svc.classes_]=d*3
        e=np.exp(full-full.max(1,keepdims=True)); return e/e.sum(1,keepdims=True)
def load_model(src):
    """Load a model from a path or bytes; accepts gzip-compressed or plain pickles."""
    data=open(src,'rb').read() if isinstance(src,str) else src
    if data[:2]==b'\x1f\x8b': data=gzip.decompress(data)
    return pickle.loads(data)
def ink(M,l):
    X=feats(l,M.get('v',1))[0]; p=M['m1'].predict_proba(X)[:,1]
    return M['m2'].predict_proba(np.c_[X,stackf(p)])[:,1]
def decode(l,w,clf,k=None,B=30,topn=3,relax=False):
    x,y,(X0,X1,y0,y1)=word_frame(l,w); H=y1-y0; span=X1-X0
    edges=np.linspace(X0-0.03*span,X1+0.03*span,B+1)
    segs=[(i,j) for i in range(B) for j in range(i+1,B+1) if relax or 0.3*H<=edges[j]-edges[i]<=1.5*H] or [(i,j) for i in range(B) for j in range(i+1,B+1)]
    F=np.array([seg_feat(x,y,w,edges[i],edges[j],y0,y1,X0,X1) for i,j in segs])
    LP=np.log(clf.predict_proba(hogf(F))+1e-9)
    inkx=np.array([(w*(np.abs(x-e)<0.04*H)).sum() for e in edges]); inkx=inkx/(inkx.max()+1e-9)
    sc={s:(LP[n].max()-1.0*inkx[s[1]]*(s[1]<B),int(LP[n].argmax()),LP[n]) for n,s in enumerate(segs)}
    res=[]
    for kk in ([k] if k else [3,4,5]):
        dp={(0,0):(0,[],[],[])}
        for n in range(1,kk+1):
            for (i,j),(s,c,lp) in sc.items():
                if (n-1,i) in dp:
                    v=dp[(n-1,i)][0]+s
                    if (n,j) not in dp or v>dp[(n,j)][0]: dp[(n,j)]=(v,dp[(n-1,i)][1]+[c],dp[(n-1,i)][2]+[lp],dp[(n-1,i)][3]+[j])
        if (kk,B) in dp:
            v,cs,lps,js=dp[(kk,B)]; res.append((v/kk,''.join(ALPHA[c] for c in cs),lps,[edges[j] for j in js[:-1]]))
    if not res:
        if not relax: return decode(l,w,clf,k,B,topn,relax=True)
        return 'A'*(k or 4),[],[]
    b=max(res,key=lambda r:r[0])
    alts=[''.join(ALPHA[i] for i in np.argsort(-lp)[:topn]) for lp in b[2]]
    return b[1],alts,b[3]
def analyze(M,l,n=1):
    """Predict one recording. Returns dict with guess, alts and a PNG (bytes)."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    w=ink(M,l); k=l.get('length')
    guess,alts,cuts=decode(l,w,M['clf'],k=k)
    s=np.array(l['samples'],float); x=s[:,2]; y=s[:,1]
    seg=np.stack([np.c_[x[:-1],y[:-1]],np.c_[x[1:],y[1:]]],1); ww=np.minimum(w[:-1],w[1:]); on=ww>0.15
    fig,axs=plt.subplots(2,1,figsize=(9,8))
    for ax,mode in zip(axs,('ink','time')):
        ax.plot(x,y,color='0.88',lw=0.7)
        if on.any():
            if mode=='ink': cols=np.zeros((on.sum(),4)); cols[:,3]=np.minimum(1,0.2+ww[on])
            else: cols=plt.cm.viridis(np.where(on)[0]/len(x)); cols[:,3]=np.minimum(1,0.2+ww[on])
            ax.add_collection(LineCollection(seg[on],colors=cols,linewidths=0.5+3*ww[on],capstyle='round'))
        ax.invert_yaxis(); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for c in cuts: axs[0].axvline(c,color='r',ls=':',lw=1)
    truth=f"   (true: {l['code']})" if l.get('code') else ''
    axs[0].set_title(f"#{n} {l.get('_file','')} {l.get('id','')} | length={k or '?'} | auto guess: {guess}{truth}",fontsize=10)
    axs[1].set_title('same ink coloured by time: purple = early, yellow = late',fontsize=9)
    plt.tight_layout(); buf=io.BytesIO(); plt.savefig(buf,format='png',dpi=80); plt.close(fig)
    return {'n':n,'file':l.get('_file',''),'id':l.get('id',''),'length':k,'guess':guess,'alts':alts,'png':buf.getvalue()}
def predict(model,paths,outdir):
    M=load_model(model) if isinstance(model,str) else model
    os.makedirs(outdir,exist_ok=True); lines=[]; results=[]
    for n,l in enumerate(read_any(paths),1):
        r=analyze(M,l,n); f=os.path.join(outdir,f'drawing_{n}.png'); open(f,'wb').write(r['png'])
        line=f"Drawing {n}: auto guess {r['guess']}   top-3 per letter: {' '.join(r['alts'])}   -> {f}"
        print(line); lines.append(line); r['path']=f; results.append(r)
    open(os.path.join(outdir,'guesses.txt'),'w').write('\n'.join(lines)+'\n')
    return results

if __name__=='__main__':
    ap=argparse.ArgumentParser(description='Invisible Ink predictor')
    sub=ap.add_subparsers(dest='cmd',required=True)
    t=sub.add_parser('train'); t.add_argument('files',nargs='+'); t.add_argument('--model',default='ink_model.pkl')
    p=sub.add_parser('predict'); p.add_argument('files',nargs='+'); p.add_argument('--model',default='ink_model.pkl'); p.add_argument('--out',default='out')
    a=ap.parse_args()
    import invisible_ink as _mod   # run via the module name so pickled classes load anywhere
    files=sum([sorted(glob.glob(f)) for f in a.files],[])
    if a.cmd=='train': _mod.train(files,a.model)
    else: _mod.predict(a.model,files,a.out)
