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
	import imgra                  # image processing with graphs
	s = x.shape                   # shape of the domain
	B = imgra.grid_incidence(*s)  # discrete gradient operator
	L = -B.T @ B                  # laplacian operator
	y = L @ x.flatten()           # laplacian of flattened data
	return y.reshape(*s)          # reshape and return


def blur_gaussian(x, σ):
	""" Gaussian blur of an image """
	from numpy.fft import fft2, ifft2, fftfreq
	from numpy import meshgrid, exp
	h,w = x.shape                           # shape of the rectangle
	p,q = meshgrid(fftfreq(w), fftfreq(h))  # build frequency abscissae
	X = fft2(x)                             # move to frequency domain
	F = exp(-σ**2 * (p**2 + q**2))          # define filter
	Y = F*X                                 # apply filter
	y = ifft2(Y).real                       # go back to spatial domain
	return y

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

def blur_riesz(x, σ):
	""" Riesz blur of an image """
	from numpy.fft import fft2, ifft2, fftfreq
	from numpy import meshgrid, exp
	h,w = x.shape                           # shape of the rectangle
	p,q = meshgrid(fftfreq(w), fftfreq(h))  # build frequency abscissae
	X = fft2(x)                             # move to frequency domain
	F = exp(-σ**2 * (p**2 + q**2))          # define filter
	Y = F*X                                 # apply filter
	y = ifft2(Y).real                       # go back to spatial domain
	return y

def __build_kernel_freq(s, σ, p, q):
	if s[0] == "g": return exp(-σ**2 * (p**2 + q**2))
	if s[0] == "c": return exp(-σ**2 * (p**2 + q**2))
	if s[0] == "l": return exp(-σ**2 * (p**2 + q**2))
	if s[0] == "r": return exp(-σ**2 * (p**2 + q**2))

def blur(x, s, σ, b="periodic"):
	""" Blur an image by the given kernel

	Args:
		x: input image
		s: name of the kernel ("gauss", "riesz", "cauchy", "disk", ...)
		σ: size parameter of the kernel (e.g. variance, radius, ...)
		b: boundary condition (default="periodic")

	Returns:
		an image of the same shape as x

	"""

	from numpy.fft import fft2, ifft2, fftfreq
	from numpy import meshgrid
	h,w = x.shape                           # shape of the rectangle
	p,q = meshgrid(fftfreq(w), fftfreq(h))  # build frequency abscissae
	X = fft2(x)                             # move to frequency domain
	F = __build_kernel_freq(s, σ, p, q)     # filter in frequency domain
	Y = F*X                                 # apply filter
	y = ifft2(Y).real                       # go back to spatial domain
	return y


# API
version = 3

__all__ = [ "sauto", "qauto", "laplacian", "blur_gaussian" ]
