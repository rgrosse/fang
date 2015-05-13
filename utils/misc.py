import numpy as np
nax = np.newaxis
import os
from PIL import Image
import progressbar
import sys
import termcolor




NEWLINE_EVERY = 50
dummy_count = [0]
def print_dot(count=None, max=None):
    print_count = (count is not None)
    if count is None:
        dummy_count[0] += 1
        count = dummy_count[0]
    sys.stdout.write('.')
    sys.stdout.flush()
    if count % NEWLINE_EVERY == 0:
        if print_count:
            if max is not None:
                sys.stdout.write(' [%d/%d]' % (count, max))
            else:
                sys.stdout.write(' [%d]' % count)
        sys.stdout.write('\n')
    elif count == max:
        sys.stdout.write('\n')
    sys.stdout.flush()


def sigmoid(t):
    return 1. / (1. + np.exp(-t))

def pbar(maxval):
    widgets = [progressbar.Percentage(), ' ', progressbar.Bar(), progressbar.ETA()]
    return progressbar.ProgressBar(widgets=widgets, maxval=maxval).start()

def str2bool(s):
    return {'True': True, 'False': False}[s]


def ensure_directory(d, trial=False):
    parts = d.split('/')
    for i in range(2, len(parts)+1):
        fname = '/'.join(parts[:i])
        if not os.path.exists(fname):
            print 'Creating', fname
            if not trial:
                try:
                    os.mkdir(fname)
                except:
                    pass

def arr2img(arr, rescale=True):
    if rescale:
        assert np.all(0. <= arr) and np.all(arr <= 1.)
        return Image.fromarray((arr*255).astype('uint8'))
    else:
        return Image.fromarray(arr.astype('uint8'))

def save_image(arr, fname):
    arr2img(arr).save(fname)

def format_time(seconds):
    if seconds < 60.:
        return '%1.1fs' % seconds
    elif seconds < 3600:
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return '%dm %ds' % (minutes, seconds)
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return '%dh %dm' % (hours, minutes)

def check_bernoulli(p, samples):
    mean = np.mean(samples)
    std = np.sqrt(p * (1. - p) / float(len(samples)))
    conf = 5. * std

    msg = 'mean = %1.5f, should be %1.5f +/- %1.5f' % (mean, p, conf)
    if p - conf < mean < p + conf:
        print msg
    else:
        print termcolor.colored(msg, 'red', attrs=['bold'])
        

def bootstrap(values, f, num=1000):
    ctr = f(values)

    samples = []
    for i in range(num):
        values_sub = values[np.random.randint(values.size, size=values.size)]
        samples.append(f(values_sub))

    over = np.percentile(samples, 97.5) - np.percentile(samples, 50)
    under = np.percentile(samples, 50) - np.percentile(samples, 2.5)
    return ctr - over, ctr + under



class gnumpy_conversion_check:
    def __init__(self, new):
        self.new = new

    def __enter__(self):
        if 'GNUMPY_IMPLICIT_CONVERSION' in os.environ:
            self.old = os.environ['GNUMPY_IMPLICIT_CONVERSION']
        else:
            self.old = None

        os.environ['GNUMPY_IMPLICIT_CONVERSION'] = self.new

    def __exit__(self, type, value, traceback):
        if self.old is not None:
            os.environ['GNUMPY_IMPLICIT_CONVERSION'] = self.old
        else:
            del os.environ['GNUMPY_IMPLICIT_CONVERSION']



class Callable:
    """Takes a function and arguments, and returns a callable which calls the function
    with those arguments. Lists the arguments in the docstring. Useful for providing
    helpful messages in nose test generators."""
    def __init__(self, fn, *args):
        self.fn = fn
        call_str = '%s.%s(%s)' % (fn.__module__, fn.__name__, ', '.join(map(repr, args)))
        self.description = '%s\n    Call: %s' % (fn.__doc__, call_str)
        self.args = args

    def __call__(self):
        return self.fn(*self.args)



def check_expectation(pred, samples):
    estimate = np.mean(samples)
    conf = 5. * np.std(samples) / np.sqrt(len(samples))

    msg = 'Predicted %1.5f; should be %1.5f +/- %1.5f' % (pred, estimate, conf)
    if estimate - conf < pred < estimate + conf:
        print msg
    else:
        print termcolor.colored(msg, 'red', attrs=['bold'])


def random_psd(D):
    A = np.random.normal(size=(2*D, D))
    return np.dot(A.T, A) / np.sqrt(2.*D)


def gauss_loglik(x, mu, Sigma):
    assert Sigma.ndim == 2
    d = Sigma.shape[0]
    return -0.5 * d * np.log(2. * np.pi) + \
           -0.5 * np.linalg.slogdet(Sigma)[1] + \
           -0.5 * np.dot(x - mu, np.linalg.solve(Sigma, x - mu))


