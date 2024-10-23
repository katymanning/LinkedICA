import numpy as np;
import DOF_eigenspectrum as DOF
#import scipy as sc
from scipy import interpolate
from scipy.optimize import fmin
#from scipy.optimize import minimize

def est_DOF_eigenspectrum(S):
	#if size(S) == len(S):
		#    % Good!
		#    assert(all(diff(S(isfinite(S)))<0))
	#elif isequal(S.shape, [len(S) len(S)]) && isequal(S,diag(diag(S))):
		#    S = diag(S);
	#elif len(S.shape) == 2:
	if S.shape[1]>S.shape[0]:
		print('error --> Matrix is wider than it is tall -- eigenspectrum method won''t work!')
	V,D = np.linalg.eig(np.dot(np.transpose(S),S))
	idx = V.argsort()[::-1]  
	s2=np.sqrt(V[idx])
	#    s2 = flipud(sqrt(eig(S'*S)));
	#    %[u S v] = svd(S,'econ');
	#    %S = diag(S);
	#    %assertalmostequal(S, s2);
	S = s2;
	#else:
	#    [~,S,~] = svd(reshape(S,[],size(S,4)),'econ');
	#    S = diag(S);
	#end
 

	#    % Use new analytic method!
	#if all(isfinite(S)) && nargin==1:
	#%        S(1:ceil(end/3)) = nan;
	#%        S(end-3:end) = nan;
	keep = np.zeros(S.shape)
	idx=np.ceil((len(S)*.25)-1); keep[idx.astype(int)] =1
	idx=np.floor((len(S)*.75)-1); keep[idx.astype(int)] =1
	#        %S(floor([1:end*.25-1, end*.25+1:end*.75-1, end*.75+1:end])) = nan; % Christian's recommended method
	#        %S(end) = nan; % Especially important to mask out the smallest eigenvalue, and sometimes the floor above doesn't quite catch it.  (484 data set ok, #'cuz it's a multiple of 4).
	noKeep=np.where(keep == 0)[0]	
	S[noKeep] = np.nan;
	#        assert(sum(isfinite(S))==2)
	#elif all(isfinite(S)):
	#        assert(isequal(r,'all'))
	
	#print 'keep going here'	
	gam = DOF.fit_eigenspectrum(np.square(S));
	dof = len(S) / gam[0];
	#dof =1
	return dof


def fit_eigenspectrum(spec):#, gam)

	#if nargin==2:
	#   assert(numel(spec)==1)
	#    gam = specest(spec, gam);
	#else:

	#% I guess a maximum-likelihood fit would be good, because with have the 
	#% PDF...
	#% But would you do that while excluding the top N points???

	#% Matlab's fminsearch allows you to set TolX, but it is always an absolute
	#% tolerance rather than relative, so it really only works sensibly when the
	#% parameters all have roughly the same scaling.  As a result, we prescale
	#% the spectrum to have a scale around 1 (rather than the 10^7 that Wooly
	#% keeps feeding me).

	prescaleSpec = np.median(spec[np.isfinite(spec)]);
	spec = spec/prescaleSpec;
	gam = np.array([.1, 1]);
	#gam=[.1,1]
	#%exitflag = 0
	#%while exitflag == 0
	#%disp 'Nonlinear fit...'
	#%[gam junk exitflag] = fminsearch(@(gam) misfit(spec,gam), gam) %[0.5 spec(end/2)])
	#%end
	#print 'keep going here'
	#gam = fminsearch(@(gam) misfit(spec,gam), gam); 
	DOF.misfit(gam,spec)
	gam = fmin(DOF.misfit,gam,args=(spec,));#method='Nelder-Mead')
	gam = np.multiply(gam, np.array([1, prescaleSpec]))  # Put the scaling back
	#gam = gam .* [1 prescaleSpec]; # Put the scaling back
	return gam

def nurange(gam,stepSize):
	out=np.arange( np.square((1-np.sqrt(gam))), np.square((1+np.sqrt(gam))) ,stepSize)
	return out

def ifeta(gam,stepSize):
	cc=DOF.nurange(gam,stepSize)
	aa=np.divide(1,(2*gam*np.pi*cc))
	bb=np.sqrt( ( cc- np.ndarray.min(cc))*( np.ndarray.max(cc)-cc )   )

	out=aa*bb
	return out

def specest(speclen, gam):
	if len(gam)==2:
	    scale = gam[1]
	    gam = gam[0]
	else:
	    scale = 1	

	if gam<=0:# gam >= 1 | scale < 0 :
		est=np.inf
	elif gam >= 1:
		est=np.inf
	elif scale <0:
		est=np.inf
	else:
		#% Note: in Johnstone[2001], gam <- 1/gam
		stepSize = .001;
		##nurange =  @(gam) (1-sqrt(gam)).^2 : stepSize : (1+sqrt(gam)).^2;
		##ifeta =  @(gam) 1./(2*gam*pi*nurange(gam)).*sqrt( (nurange(gam)-min(nurange(gam))).*(max(nurange(gam))-nurange(gam)) );

		nu = DOF.nurange(gam,stepSize) #nurange(gam);
		tmp=DOF.ifeta(gam,stepSize)
		cif = np.cumsum( tmp ) *stepSize;

		xranges = np.linspace(0,1,speclen);
		cif[-1] = 1;
		#assert(all(isfinite([cif(:);nu(:);xrange(:)])))
		#est = interp1(cif, nu, xrange);
#print 'keep going here'
		est=interpolate.interp1d(cif,nu,kind='linear')(xranges)
		#%assert(rms(est-est2)/rms(est)<1e-3))
		idx = est.argsort()[::-1]
		#est = flipud(est(:)) * scale;
		est=est[idx]*scale
	return est


def misfit(gam,spec):
	est = DOF.specest(len(spec), gam);
	ssd = np.square(spec-est);
	ssd = np.sum(ssd[np.isfinite(spec)]);
	return ssd


