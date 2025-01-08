def ntiply(x, n=4):
	""" zoom-in an image by pixel replication """
	y = x.repeat(n,axis=0).repeat(n,axis=1)
	return y


# function to show a signed image with red-white-blue palette
def sauto(x, q=0.995):
	""" RGB rendering of a signed scalar image using a divergent palette """
	from numpy import clip, fabs, dstack, nanquantile, nan_to_num
	s = nanquantile(fabs(x), q)    # find saturation quantile
	r = 1 - clip(x/s, 0, 1)        # red component
	g = 1 - clip(fabs(x/s), 0, 1)  # green
	b = 1 + clip(x/s, -1, 0)       # blue
	c = dstack([r, g, b])          # color
	c = clip(c, 0, 1)              # saturate color into [0,1]
	c = nan_to_num(c, nan=0.5)     # set nans to gray
	c = (255*c).astype(int)        # rescale and quantize
	return c


def qauto(x, q=0.995, i=True, n=True):
	"""
	quantize a floating-point image to 8 bits per channel

	Args:
		x: input image
		q: saturation quantile (default q=0.995)
		i: whether to treat all channels independently (default=True)
		n: whether to paint NaNs in blue (default=True)

	Returns:
		a 8-bit image (rgb if n=True, otherwise grayscale)

	"""
	if i and len(x.shape)==3 and x.shape[2]>1:
		from numpy import dstack
		return dstack([
			qauto(x[:,:,c], q, i=False, n=False)
			for c in range(x.shape[2])
			])
	from numpy import nanquantile, clip, uint8
	s = nanquantile(x, 1-q)          # lower saturation quantile
	S = nanquantile(x, q)            # upper saturation quantile
	y = clip((x - s)/(S - s), 0, 1)  # saturate values to [0,1]
	if n and (len(y.shape)==2 or (len(y.shape)==3 and y.shape[2]==1)):
		from numpy import isnan, dstack
		r = 1 * y
		g = 1 * y
		b = 1 * y
		r[isnan(x)] = 0
		g[isnan(x)] = 0
		b[isnan(x)] = 0.4
		y = dstack([r, g, b])
	else:
		from numpy import nan_to_num
		y = nan_to_num(y, nan=0.5) # set nans to middle gray
	y = (255*y).astype(uint8)          # rescale and quantize
	return y


def laplacian(x):
	""" Compute the five-point laplacian of an image """
	if len(x.shape)==3:
		from numpy import dstack as d
		return d([ laplacian(x[:,:,c]) for c in range(x.shape[2]) ])
	import imgra                  # image processing with graphs
	s = x.shape                   # shape of the domain
	B = imgra.grid_incidence(*s)  # discrete gradient operator
	L = -B.T @ B                  # laplacian operator
	y = L @ x.flatten()           # laplacian of flattened data
	return y.reshape(*s)          # reshape and return


def gradient(x):
	""" Compute the gradient by forward-differences """
	if len(x.shape)==3:
		from numpy import dstack as d
		return d([ gradient(x[:,:,c]) for c in range(x.shape[2]) ])
	import imgra                   # image processing with graphs
	h,w = x.shape                  # shape of the domain
	B = imgra.grid_incidence(h,w)  # discrete gradient operator
	g = B @ x.flatten()            # gradient of flattened data
	G = 0 * x[:,:,None].repeat(2,axis=2)
	G[:h,:w-1,0] = g[:h*(w-1)].reshape(h,w-1)
	G[:h-1,:w,1] = g[h*(w-1):].reshape(h-1,w)
	return G


def divergence(x):
	""" Compute the divergence by backward-differences """
	if x.shape[2] != 2:
		from numpy import dstack
		return dstack([ divergence(x[:,:,2*c:2*c+2])
		                for c in range(x.shape[2]//2) ])
	assert 2 == x.shape[2]
	h,w,_ = x.shape
	import imgra
	B = imgra.grid_incidence(h,w)
	f = x[:,:-1,0].flatten()
	g = x[:-1,:,1].flatten()
	from numpy import hstack
	return ( B.T @ hstack([-f,-g]) ).reshape(h,w)



def viewdft(x):
	""" display the DFT of an image in an intuitive way """

	if len(x.shape)==3:
		from numpy import dstack as d
		return d([ viewdft(x[:,:,c]) for c in range(x.shape[2]) ])

	from numpy import abs, log
	from numpy.fft import fft2, fftshift
	X = fft2(x)
	v = qauto(log(1+abs(fftshift(X))))[:,:,0]
	return v




def ppsmooth(I):
	""" Compute the periodic+smooth decomposition of an image """
	# NOTE: implementation by Jacob Kimmel of Moisan's algorithm
	# https://github.com/jacobkimmel/ps_decomp

	if len(I.shape)==3:
		from numpy import dstack as d
		return d([ ppsmooth(I[:,:,c]) for c in range(I.shape[2]) ])

	def v2s(V):
		from numpy import pi as π, arange, cos, errstate
		M, N = V.shape
		q = arange(M).reshape(M, 1).astype(V.dtype)
		r = arange(N).reshape(1, N).astype(V.dtype)
		d = (2*cos(2*π*q/M) + 2*cos(2*π*r/N) - 4)
		with errstate(all="ignore"):
			s = V / d
		s[0, 0] = 0
		return s

	def u2v(u):
		v = 0 * u
		v[ 0, :]  = u[-1, :] - u[ 0, :]
		v[-1, :]  = u[ 0, :] - u[-1, :]
		v[ :, 0] += u[ :,-1] - u[ :, 0]
		v[ :,-1] += u[ :, 0] - u[ :,-1]
		return v

	from numpy.fft import fft2, ifft2
	u = I
	v = u2v(u)
	V = fft2(v)
	S = v2s(V)
	s = ifft2(S).real
	p = u - s
	return p #, s


#def blur_gaussian(x, σ):
#	""" Gaussian blur of an image """
#	from numpy.fft import fft2, ifft2, fftfreq
#	from numpy import meshgrid, exp
#	h,w = x.shape                           # shape of the rectangle
#	p,q = meshgrid(fftfreq(w), fftfreq(h))  # build frequency abscissae
#	X = fft2(x)                             # move to frequency domain
#	F = exp(-σ**2 * (p**2 + q**2))          # define filter
#	Y = F*X                                 # apply filter
#	y = ifft2(Y).real                       # go back to spatial domain
#	return y

#def blur_laplace(x, σ):
#	""" Laplacian blur of an image """
#	from numpy.fft import fft2, ifft2, fftfreq
#	from numpy import meshgrid, exp
#	h,w = x.shape                           # shape of the rectangle
#	p,q = meshgrid(fftfreq(w), fftfreq(h))  # build frequency abscissae
#	X = fft2(x)                             # move to frequency domain
#	F = exp(-σ**2 * (p**2 + q**2))          # define filter
#	Y = F*X                                 # apply filter
#	y = ifft2(Y).real                       # go back to spatial domain
#	return y

#def blur_riesz(x, σ):
#	""" Riesz blur of an image """
#	from numpy.fft import fft2, ifft2, fftfreq
#	from numpy import meshgrid, exp
#	h,w = x.shape                           # shape of the rectangle
#	p,q = meshgrid(fftfreq(w), fftfreq(h))  # build frequency abscissae
#	X = fft2(x)                             # move to frequency domain
#	F = exp(-σ**2 * (p**2 + q**2))          # define filter
#	Y = F*X                                 # apply filter
#	y = ifft2(Y).real                       # go back to spatial domain
#	return y

def __build_kernel_freq(s, σ, p, q):
	from numpy import exp, sinc, fabs, fmax
	from numpy import pi as π
	r2 = p**2 + q**2
	if s[0] == "g": return exp(-2 * π**2 * σ**2 * r2)         # gauss
	if s[0] == "l": return 1/(1 + σ*r2)                       # laplace
	if s[0] == "c": return exp(-σ * r2**0.5)                  # cauchy
	if s[0] == "D": return sinc(2 * σ * r2**0.5)              # Disk
	if s[0] == "S": return sinc(2*σ*fabs(p)) * sinc(2*σ*fabs(q))  # Square
	if s[0] == "d":                                           # disk
		from numpy.fft import fft2
		P = p.shape[1] * p
		Q = p.shape[0] * q
		F = fft2( P**2 + Q**2 < σ**2 )
		F[0,0] = 1
		return F
	if s[0] == "s":                                           # square
		from numpy.fft import fft2
		P = p.shape[1] * p
		Q = p.shape[0] * q
		F = fft2( fmax(fabs(P),fabs(Q)) < σ )
		F[0,0] = 1
		return F
	if s[0] == "z":                                           # zquare
		from numpy.fft import fft2
		P = p.shape[1] * p
		Q = p.shape[0] * q
		F = fft2( fabs(P)+fabs(Q) < σ )
		F[0,0] = 1
		return F
	if s[0] == "r":                                           # riesz
		r2[0,0] = 1
		F = 1/r2**(σ/2)
		F[0,0] = 0
		return F

def blur(x, k, σ, b="periodic"):
	""" Blur an image by the given kernel

	Args:
		x: input image
		k: name of the kernel ("gauss", "riesz", "cauchy", "disk", ...)
		σ: size parameter of the kernel (e.g. variance, radius, ...)
		b: boundary condition (default="periodic")

	Returns:
		an image of the same shape as x

	"""

	# for multidimensional pixels, blur each channel separately
	if len(x.shape)==3:
		from numpy import dstack as d
		return d([ blur(x[:,:,c],k,σ,b) for c in range(x.shape[2]) ])

	# apply boundary condition in the case d=1
	h,w = x.shape                           # shape of the rectangle
	if b == "zero": b = "constant"
	if b[0] != "p":
		from numpy import pad
		return blur(pad(x,((0,w),(0,h)),mode=b),k,σ,b="p")[:h,:w]

	# base case with d=1 and periodic boundary
	from numpy.fft import fft2, ifft2, fftfreq
	from numpy import meshgrid
	p,q = meshgrid(fftfreq(w), fftfreq(h))  # build frequency abscissae
	X = fft2(x)                             # move to frequency domain
	F = __build_kernel_freq(k, σ, p, q)     # filter in frequency domain
	Y = F*X                                 # apply filter
	y = ifft2(Y).real                       # go back to spatial domain
	return y


def plambda(x, e):
	""" Apply an expression to an image """
	exec(f"def f(x): return {e}", globals())
	from numpy import vectorize as v
	return v(f)(x)



# visible API
__all__ = [ "sauto", "qauto", "laplacian", "gradient", "divergence",
	   "blur", "ntiply", "ppsmooth", "plambda" ]


# cli interfaces to the above functions
if __name__ == "__main__":
	from sys import argv as v
	def pick_option(o, d):
		if int == type(o): return v[o]
		return type(d)(v[v.index(o)+1]) if o in v else d
	if len(v) < 2 or v[1] not in __all__:
		print(f"usage:\n\tymscript {{{'|'.join(__all__)}}}")
		exit(0)
	import iio
	i = pick_option("-i", "-")
	o = pick_option("-o", "-")
	x = iio.read(i)
	if "blur" == v[1]:
		k = pick_option("-k", "gaussian")
		s = pick_option("-s", 3.0)
		b = pick_option("-b", "periodic")
		y = blur(x, k, s, b)
	if "laplacian" == v[1]:
		y = laplacian(x)
	if "gradient" == v[1]:
		y = gradient(x)
	if "divergence" == v[1]:
		y = divergence(x)
	if "qauto" == v[1]:
		q = pick_option("-q", 0.995)
		s = pick_option("-s", True)
		n = pick_option("-n", True)
		y = qauto(x, q, s, n)
	if "sauto" == v[1]:
		q = pick_option("-q", 0.995)
		x = x.squeeze()
		if len(x.shape)==3:
			x = x[:,:,0]
		y = sauto(x, q)
	if "ntiply" == v[1]:
		q = pick_option("-n", 4)
		y = ntiply(x, q)
	if "ppsmooth" == v[1]:
		y = ppsmooth(x)
	if "plambda" == v[1]:
		e = pick_option(2, "x")
		y = plambda(x, e)
	iio.write(o, y)



# API
version = 10
