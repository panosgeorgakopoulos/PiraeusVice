import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import seaborn as sns
from scipy.stats import norm, chi2
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from scipy.stats import multivariate_normal
import warnings; warnings.filterwarnings('ignore')
import os

np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(file))

OUT = os.path.join(BASE_DIR, 'outputs') + os.sep
DATA = os.path.join(BASE_DIR, 'crimes.csv')

os.makedirs(OUT, exist_ok=True)
CONT = ['hour_float','latitude','longitude','victim_age','temp_c','humidity','dist_precinct_km','pop_density']
CATS = ['weapon_code','scene_type','weather','vic_gender']
S = 8

df = pd.read_csv(DATA)
tr = df[df['split']=='TRAIN'].copy()
vl = df[df['split']=='VAL'].copy()
te = df[df['split']=='TEST'].copy()

def ohe(data):
    d = pd.get_dummies(data[CATS].astype(str), columns=CATS)
    return pd.concat([data[CONT].reset_index(drop=True), d.reset_index(drop=True)], axis=1)

Xtr0 = ohe(tr); Xvl0 = ohe(vl); Xte0 = ohe(te)
cols = Xtr0.columns.tolist()
Xvl0 = Xvl0.reindex(columns=cols, fill_value=0)
Xte0 = Xte0.reindex(columns=cols, fill_value=0)

Xtr = Xtr0.values.astype(float); Xvl = Xvl0.values.astype(float); Xte = Xte0.values.astype(float)
ytr = tr['killer_id'].values; yvl = vl['killer_id'].values
Xtr_c = tr[CONT].values.astype(float)
Xvl_c = vl[CONT].values.astype(float)
Xte_c = te[CONT].values.astype(float)

ks = np.arange(1, S+1)
C10 = plt.cm.tab10(np.linspace(0,0.8,S))
KL  = [f'K{k}' for k in ks]
print('Loaded:', df.shape)

# ── Q1 ──
print('\n=== Q1 ===')
tv = pd.concat([tr,vl], ignore_index=True)
fig,axes = plt.subplots(2,2,figsize=(12,8))
fig.suptitle('Q1: Feature Histograms (TRAIN+VAL)',fontsize=14,fontweight='bold')
for ax,c in zip(axes.flatten(),['hour_float','victim_age','latitude','longitude']):
    ax.hist(tv[c],bins=40,color='steelblue',edgecolor='white',alpha=0.8)
    ax.set_title(c); ax.set_xlabel(c); ax.set_ylabel('Count')
plt.tight_layout(); plt.savefig(OUT+'q1_histograms.png',dpi=130,bbox_inches='tight'); plt.close()

h = tv['hour_float'].values; mu_h,std_h = h.mean(),h.std()
gmm = GaussianMixture(n_components=3,random_state=42).fit(h.reshape(-1,1))
gm,gs,gw = gmm.means_.flatten(), np.sqrt(gmm.covariances_.flatten()), gmm.weights_
fig,ax=plt.subplots(figsize=(10,5))
ax.hist(h,bins=50,density=True,color='lightblue',edgecolor='white',alpha=0.7,label='Data')
xr=np.linspace(0,24,500)
ax.plot(xr,norm.pdf(xr,mu_h,std_h),'r-',lw=2.5,label=f'Gaussian μ={mu_h:.2f} σ={std_h:.2f}')
gd=sum(w*norm.pdf(xr,m,s) for w,m,s in zip(gw,gm,gs))
ax.plot(xr,gd,'g-',lw=2.5,label='3-comp GMM')
for i,(w,m,s) in enumerate(zip(gw,gm,gs)):
    ax.plot(xr,w*norm.pdf(xr,m,s),'--',lw=1.4,alpha=0.6,label=f'Comp{i+1} w={w:.2f} μ={m:.2f}')
ax.set_title('Q1: hour_float GMM',fontweight='bold'); ax.set_xlabel('hour_float'); ax.set_ylabel('Density')
ax.legend(fontsize=8); ax.set_xlim(0,24)
plt.tight_layout(); plt.savefig(OUT+'q1_gmm_fit.png',dpi=130,bbox_inches='tight'); plt.close()

fig,ax=plt.subplots(figsize=(10,5))
ax.scatter(tv['hour_float'],tv['latitude'],s=3,alpha=0.3,c='navy')
ax.set_title('Q1: hour_float vs latitude',fontweight='bold'); ax.set_xlabel('hour_float'); ax.set_ylabel('latitude')
plt.tight_layout(); plt.savefig(OUT+'q1_2d_exploration.png',dpi=130,bbox_inches='tight'); plt.close()
print('  Q1 done')

# ── Q2 ──
print('=== Q2 ===')
mu_k,Sk={},{}
for k in ks:
    idx=np.where(ytr==k)[0]; Xk=Xtr_c[idx]
    mu=Xk.mean(0); d=Xk-mu; Sk[k]=(d.T@d)/len(idx); mu_k[k]=mu
    ll=sum(multivariate_normal.logpdf(x,mu,Sk[k]+1e-6*np.eye(8)) for x in Xk)
    ll2=sum(multivariate_normal.logpdf(x,Xk.mean(0),np.cov(Xk.T,bias=True)+1e-6*np.eye(8)) for x in Xk)
    print(f'  K{k}: LL_manual={ll:.1f} LL_lib={ll2:.1f} diff={abs(ll-ll2):.5f}')

short=['hour','lat','lon','age','temp','hum','dist','pop']
fig,axes=plt.subplots(2,4,figsize=(18,8))
fig.suptitle('Q2: Correlation Heatmaps',fontsize=14,fontweight='bold')
for ax,k in zip(axes.flatten(),ks):
    d=np.sqrt(np.diag(Sk[k])); corr=Sk[k]/(d[:,None]*d[None,:])
    im=ax.imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r')
    ax.set_title(f'K{k}',fontweight='bold',fontsize=10)
    ax.set_xticks(range(8)); ax.set_yticks(range(8))
    ax.set_xticklabels(short,fontsize=7,rotation=45); ax.set_yticklabels(short,fontsize=7)
    plt.colorbar(im,ax=ax,shrink=0.7)
plt.tight_layout(); plt.savefig(OUT+'q2_covariance_heatmaps.png',dpi=130,bbox_inches='tight'); plt.close()

def draw_ell(mu2,S2,ax,col):
    ev,evec=np.linalg.eigh(S2); ang=np.degrees(np.arctan2(*evec[:,1][::-1]))
    sc=np.sqrt(chi2.ppf(0.95,df=2)); w,hh=2*sc*np.sqrt(np.abs(ev))
    ax.add_patch(Ellipse(xy=mu2,width=w,height=hh,angle=ang,edgecolor=col,fc='none',lw=2))

fig,axes=plt.subplots(1,2,figsize=(14,6))
fig.suptitle('Q2: 95% Confidence Ellipses',fontsize=13,fontweight='bold')
for k in ks:
    idx=np.where(ytr==k)[0]; col=C10[k-1]
    for ax,(i1,i2),xl,yl in [(axes[0],(1,2),'latitude','longitude'),(axes[1],(1,0),'latitude','hour_float')]:
        pts=Xtr_c[idx][:,[i1,i2]]; ax.scatter(pts[:,0],pts[:,1],s=4,alpha=0.3,color=col)
        draw_ell(mu_k[k][[i1,i2]],Sk[k][np.ix_([i1,i2],[i1,i2])],ax,col)
for ax,(xl,yl) in zip(axes,[('latitude','longitude'),('latitude','hour_float')]):
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(f'{xl} vs {yl}')
    ax.legend(handles=[mpatches.Patch(color=C10[k-1],label=f'K{k}') for k in ks],fontsize=7,ncol=2)
plt.tight_layout(); plt.savefig(OUT+'q2_ellipses.png',dpi=130,bbox_inches='tight'); plt.close()
print('  Q2 done')

# ── Q3 ──
print('=== Q3 ===')
Nk={k:int(np.sum(ytr==k)) for k in ks}; pi={k:Nk[k]/len(ytr) for k in ks}; REG=1e-4

def bayes(X,rp=False):
    lp=np.zeros((len(X),S))
    for k in ks:
        Sr=Sk[k]+REG*np.eye(8); _,ld=np.linalg.slogdet(Sr); iS=np.linalg.inv(Sr)
        d=X-mu_k[k]; mah=np.einsum('nd,dd,nd->n',d,iS,d)
        lp[:,k-1]=np.log(pi[k])-0.5*ld-0.5*mah
    post=np.exp(lp-logsumexp(lp,axis=1,keepdims=True)); pr=post.argmax(1)+1
    return (pr,post) if rp else pr

vl_pb,vl_prb=bayes(Xvl_c,rp=True)
tr_pb=bayes(Xtr_c)
vl_ab=accuracy_score(yvl,vl_pb); tr_ab=accuracy_score(ytr,tr_pb)
print(f'  Bayes TRAIN={tr_ab:.4f} VAL={vl_ab:.4f}')

fig,ax=plt.subplots(figsize=(8,6))
sns.heatmap(confusion_matrix(yvl,vl_pb,labels=ks),annot=True,fmt='d',cmap='Blues',
            xticklabels=KL,yticklabels=KL,ax=ax)
ax.set_title(f'Q3 Bayes CM (VAL={vl_ab:.3f})',fontweight='bold')
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
plt.tight_layout(); plt.savefig(OUT+'q3_confusion_matrix.png',dpi=130,bbox_inches='tight'); plt.close()

sc2=StandardScaler(); Ztr2=PCA(n_components=2,random_state=42).fit_transform(sc2.fit_transform(Xtr_c))
pca2=PCA(n_components=2,random_state=42).fit(sc2.transform(Xtr_c))
Ztr2=pca2.transform(sc2.transform(Xtr_c)); Zvl2=pca2.transform(sc2.transform(Xvl_c))
x1,x2=Ztr2[:,0].min()-1,Ztr2[:,0].max()+1; y1,y2=Ztr2[:,1].min()-1,Ztr2[:,1].max()+1
xx,yy=np.meshgrid(np.linspace(x1,x2,120),np.linspace(y1,y2,120))
gc=sc2.inverse_transform(pca2.inverse_transform(np.c_[xx.ravel(),yy.ravel()]))
gpb=bayes(gc)

fig,ax=plt.subplots(figsize=(10,7))
ax.contourf(xx,yy,gpb.reshape(xx.shape),alpha=0.2,cmap='tab10',levels=np.arange(0.5,S+1.5))
for k in ks:
    idx=ytr==k; ax.scatter(Ztr2[idx,0],Ztr2[idx,1],s=12,color=C10[k-1],label=f'K{k}',alpha=0.7)
ax.set_title('Q3: Bayes Decision Regions (PCA-2D)',fontweight='bold')
ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.legend(fontsize=8,markerscale=2)
plt.tight_layout(); plt.savefig(OUT+'q3_decision_regions.png',dpi=130,bbox_inches='tight'); plt.close()
print('  Q3 done')

# ── Q4 ──
print('=== Q4 ===')
sl=StandardScaler(); Xtr_l=sl.fit_transform(Xtr); Xvl_l=sl.transform(Xvl); Xte_l=sl.transform(Xte)
lin=LogisticRegression(max_iter=2000,C=1.0,solver='lbfgs',random_state=42)
lin.fit(Xtr_l,ytr)
vl_pl=lin.predict(Xvl_l); vl_pl_prob=lin.predict_proba(Xvl_l)
vl_al=accuracy_score(yvl,vl_pl); tr_al=accuracy_score(ytr,lin.predict(Xtr_l))
print(f'  Linear TRAIN={tr_al:.4f} VAL={vl_al:.4f}')

fig,ax=plt.subplots(figsize=(8,6))
sns.heatmap(confusion_matrix(yvl,vl_pl,labels=ks),annot=True,fmt='d',cmap='Oranges',
            xticklabels=KL,yticklabels=KL,ax=ax)
ax.set_title(f'Q4 Linear CM (VAL={vl_al:.3f})',fontweight='bold')
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
plt.tight_layout(); plt.savefig(OUT+'q4_confusion_matrix.png',dpi=130,bbox_inches='tight'); plt.close()

sl2d=StandardScaler(); Zt2s=sl2d.fit_transform(Ztr2)
lin2d=LogisticRegression(max_iter=500,C=1.0,solver='lbfgs')
lin2d.fit(Zt2s,ytr)
gpl=lin2d.predict(sl2d.transform(np.c_[xx.ravel(),yy.ravel()]))

fig,axes=plt.subplots(1,2,figsize=(14,6))
for ax,gp,t in zip(axes,[gpb,gpl],['Q3 Bayes','Q4 Linear']):
    ax.contourf(xx,yy,gp.reshape(xx.shape),alpha=0.2,cmap='tab10',levels=np.arange(0.5,S+1.5))
    for k in ks:
        idx=ytr==k; ax.scatter(Ztr2[idx,0],Ztr2[idx,1],s=10,color=C10[k-1],label=f'K{k}',alpha=0.6)
    ax.set_title(t,fontweight='bold'); ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.legend(fontsize=6,markerscale=2)
plt.tight_layout(); plt.savefig(OUT+'q4_bayes_vs_linear.png',dpi=130,bbox_inches='tight'); plt.close()
print('  Q4 done')

# ── Q5 ──
print('=== Q5 ===')
svm=SVC(kernel='rbf',C=10.0,gamma='scale',probability=True,decision_function_shape='ovr',random_state=42)
svm.fit(Xtr_l,ytr)
vl_ps=svm.predict(Xvl_l); vl_ps_prob=svm.predict_proba(Xvl_l)
vl_as=accuracy_score(yvl,vl_ps); tr_as=accuracy_score(ytr,svm.predict(Xtr_l))
print(f'  SVM TRAIN={tr_as:.4f} VAL={vl_as:.4f}')

fig,ax=plt.subplots(figsize=(8,6))
sns.heatmap(confusion_matrix(yvl,vl_ps,labels=ks),annot=True,fmt='d',cmap='Greens',
            xticklabels=KL,yticklabels=KL,ax=ax)
ax.set_title(f'Q5 SVM CM (VAL={vl_as:.3f})',fontweight='bold')
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
plt.tight_layout(); plt.savefig(OUT+'q5_confusion_matrix.png',dpi=130,bbox_inches='tight'); plt.close()

svm2d=SVC(kernel='rbf',C=10.0,gamma='scale',probability=True,random_state=42)
svm2d.fit(Zt2s,ytr)
gps=svm2d.predict(sl2d.transform(np.c_[xx.ravel(),yy.ravel()]))

fig,ax=plt.subplots(figsize=(10,7))
ax.contourf(xx,yy,gps.reshape(xx.shape),alpha=0.2,cmap='tab10',levels=np.arange(0.5,S+1.5))
for k in ks:
    idx=ytr==k; ax.scatter(Ztr2[idx,0],Ztr2[idx,1],s=10,color=C10[k-1],label=f'K{k}',alpha=0.6)
sv_o=sl2d.inverse_transform(Zt2s[svm2d.support_])
ax.scatter(sv_o[:,0],sv_o[:,1],s=80,facecolors='none',edgecolors='k',lw=1.5,label='SVs',zorder=5)
ax.set_title('Q5: SVM + Support Vectors (PCA-2D)',fontweight='bold')
ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.legend(fontsize=7,markerscale=2)
plt.tight_layout(); plt.savefig(OUT+'q5_svm_decision.png',dpi=130,bbox_inches='tight'); plt.close()
print('  Q5 done')

# ── Q6 ──
print('=== Q6 ===')
mlp=MLPClassifier(hidden_layer_sizes=(128,64,32),activation='relu',solver='adam',
                  alpha=1e-3,batch_size=256,learning_rate_init=1e-3,max_iter=300,
                  early_stopping=True,validation_fraction=0.1,random_state=42,verbose=False)
mlp.fit(Xtr_l,ytr)
vl_pm=mlp.predict(Xvl_l); vl_pm_prob=mlp.predict_proba(Xvl_l)
vl_am=accuracy_score(yvl,vl_pm); tr_am=accuracy_score(ytr,mlp.predict(Xtr_l))
print(f'  MLP TRAIN={tr_am:.4f} VAL={vl_am:.4f}')

fig,ax=plt.subplots(figsize=(8,6))
sns.heatmap(confusion_matrix(yvl,vl_pm,labels=ks),annot=True,fmt='d',cmap='Purples',
            xticklabels=KL,yticklabels=KL,ax=ax)
ax.set_title(f'Q6 MLP CM (VAL={vl_am:.3f})',fontweight='bold')
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
plt.tight_layout(); plt.savefig(OUT+'q6_confusion_matrix.png',dpi=130,bbox_inches='tight'); plt.close()

print('  Computing permutation importance...')
imp=[]
for j in range(Xvl_l.shape[1]):
    Xp=Xvl_l.copy(); Xp[:,j]=np.random.permutation(Xp[:,j])
    imp.append(vl_am - accuracy_score(yvl,mlp.predict(Xp)))
imp=np.array(imp); top10=np.argsort(imp)[::-1][:10]
bc=['salmon' if cols[i] in CONT else 'steelblue' for i in top10]
fig,ax=plt.subplots(figsize=(10,5))
ax.bar(range(10),imp[top10],color=bc,edgecolor='white')
ax.set_xticks(range(10)); ax.set_xticklabels([cols[i] for i in top10],rotation=40,ha='right',fontsize=9)
ax.set_title('Q6: Permutation Feature Importance (Top 10)',fontweight='bold'); ax.set_ylabel('Accuracy Drop')
ax.legend(handles=[mpatches.Patch(color='salmon',label='Continuous'),mpatches.Patch(color='steelblue',label='Categorical')],fontsize=9)
plt.tight_layout(); plt.savefig(OUT+'q6_feature_importance.png',dpi=130,bbox_inches='tight'); plt.close()
print('  Top5:',[cols[i] for i in top10[:5]])
print('  Q6 done')

# ── Q7 ──
print('=== Q7 ===')
sf=StandardScaler(); Xtr_f=sf.fit_transform(Xtr); Xvl_f=sf.transform(Xvl); Xte_f=sf.transform(Xte)
pca=PCA(random_state=42); pca.fit(Xtr_f)
ev=pca.explained_variance_; cv=np.cumsum(pca.explained_variance_ratio_)

fig,axes=plt.subplots(1,2,figsize=(13,5))
axes[0].plot(range(1,len(ev)+1),ev,'bo-',markersize=5,lw=1.5)
axes[0].axvline(x=10,color='r',ls='--',label='~10')
axes[0].set_title('Q7: Scree Plot',fontweight='bold'); axes[0].set_xlabel('Component'); axes[0].set_ylabel('Eigenvalue'); axes[0].legend()
axes[1].plot(range(1,len(cv)+1),cv*100,'g-o',markersize=4)
axes[1].axhline(y=90,color='r',ls='--',label='90%')
axes[1].set_title('Q7: Cumulative Variance',fontweight='bold'); axes[1].set_xlabel('# Components'); axes[1].set_ylabel('Var %'); axes[1].legend()
plt.tight_layout(); plt.savefig(OUT+'q7_pca_scree.png',dpi=130,bbox_inches='tight'); plt.close()

m=int(np.argmax(cv>=0.90))+1; print(f'  m={m} for 90% variance, first-10={cv[9]*100:.1f}%')
pcam=PCA(n_components=m,random_state=42)
Ztr_m=pcam.fit_transform(Xtr_f); Zvl_m=pcam.transform(Xvl_f); Zte_m=pcam.transform(Xte_f)
Zvl2=Zvl_m[:,:2]

fig,ax=plt.subplots(figsize=(10,7))
for k in ks:
    idx=vl_ps==k
    if idx.any(): ax.scatter(Zvl2[idx,0],Zvl2[idx,1],s=15,color=C10[k-1],label=f'K{k}',alpha=0.7)
ax.set_title('Q7: VAL PC1–PC2 (SVM labels)',fontweight='bold'); ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
ax.legend(fontsize=8,markerscale=2)
plt.tight_layout(); plt.savefig(OUT+'q7_pca_scatter.png',dpi=130,bbox_inches='tight'); plt.close()
print('  Q7 done')

# ── Q8 ──
print('=== Q8 ===')
km=KMeans(n_clusters=S,init='k-means++',n_init=20,random_state=42,max_iter=500)
km.fit(Ztr_m); tr_cl=km.labels_
gmap={}
for q in range(S):
    iq=np.where(tr_cl==q)[0]; kc={k:int(np.sum(ytr[iq]==k)) for k in ks}; gmap[q]=max(kc,key=kc.get)
print(f'  Cluster->Killer: {gmap}')

vl_cl=km.predict(Zvl_m); vl_pkm=np.array([gmap[c] for c in vl_cl])
vl_akm=accuracy_score(yvl,vl_pkm); print(f'  K-Means VAL={vl_akm:.4f}')
te_cl=km.predict(Zte_m); te_pkm=np.array([gmap[c] for c in te_cl])
Zte2=Zte_m[:,:2]

fig,axes=plt.subplots(1,2,figsize=(14,6))
fig.suptitle('Q8: K-Means Clustering',fontsize=13,fontweight='bold')
for k in ks:
    iv=vl_pkm==k
    if iv.any(): axes[0].scatter(Zvl2[iv,0],Zvl2[iv,1],s=15,color=C10[k-1],label=f'K{k}',alpha=0.7)
axes[0].set_title(f'VAL (Acc={vl_akm:.3f})'); axes[0].set_xlabel('PC1'); axes[0].set_ylabel('PC2'); axes[0].legend(fontsize=7,markerscale=2)
for k in ks:
    it=te_pkm==k
    if it.any(): axes[1].scatter(Zte2[it,0],Zte2[it,1],s=15,color=C10[k-1],label=f'K{k}',alpha=0.7)
axes[1].set_title('TEST Predictions'); axes[1].set_xlabel('PC1'); axes[1].set_ylabel('PC2'); axes[1].legend(fontsize=7,markerscale=2)
plt.tight_layout(); plt.savefig(OUT+'q8_kmeans.png',dpi=130,bbox_inches='tight'); plt.close()
print('  Q8 done')

# ── SUBMISSION ──
print('\n=== Generating submission.csv ===')
all_oh=ohe(df).reindex(columns=cols,fill_value=0).values.astype(float)
all_Xl=sl.transform(all_oh); all_pr=mlp.predict(all_Xl); all_pb=mlp.predict_proba(all_Xl)
sub=pd.DataFrame({'incident_id':df['incident_id'].values,'predicted_killer':all_pr})
for ki,k in enumerate(mlp.classes_): sub[f'p_killer_{k}']=all_pb[:,ki]
sub.to_csv(OUT+'submission.csv',index=False)
print(f'  submission.csv: {len(sub)} rows')

# ── COMPARISON CHART ──
mnms=['Gaussian\nBayes','Linear','SVM\n(RBF)','MLP','K-Means']
accs=[vl_ab,vl_al,vl_as,vl_am,vl_akm]
fig,ax=plt.subplots(figsize=(9,5))
bars=ax.bar(mnms,accs,color=['#4C72B0','#DD8452','#55A868','#C44E52','#8172B2'],edgecolor='white',width=0.6)
ax.set_ylim(0,1); ax.set_ylabel('VAL Accuracy',fontsize=11)
ax.set_title('Model Comparison — VAL Accuracy',fontsize=12,fontweight='bold')
for bar,acc in zip(bars,accs):
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.01,f'{acc:.3f}',ha='center',va='bottom',fontsize=10,fontweight='bold')
ax.axhline(y=1/S,color='grey',ls='--',label=f'Random 1/{S}={1/S:.2f}'); ax.legend()
plt.tight_layout(); plt.savefig(OUT+'model_comparison.png',dpi=130,bbox_inches='tight'); plt.close()

print('\n=== FINAL SUMMARY ===')
for mn,ac in zip(mnms,accs): print(f'  {mn.replace(chr(10)," "):20s}: VAL={ac:.4f}')
print('\nAll done!')