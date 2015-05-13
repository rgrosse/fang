import copy
import gnumpy as gnp
import numpy as np


def copy2(a):
    try:
        return a.copy()
    except:
        return a

def isscalar(a):
    return np.isscalar(a) or (isinstance(a, np.ndarray) and a.shape == ())



class Struct:
    mutable = True

    def __init__(self, *args, **kwargs):
        for f, a in zip(self.fields, args):
            if isinstance(a, np.ndarray):
                a = a.copy()
            setattr(self, f, a)
            if not self.mutable and isinstance(a, np.ndarray):
                a.flags.writeable = False
        for f, v in kwargs.items():
            if isinstance(v, np.ndarray):
                v = v.copy()
            setattr(self, f, v)
            if not self.mutable and isinstance(v, np.ndarray):
                v.flags.writeable = False

        for f in self.fields:
            if not hasattr(self, f):
                raise RuntimeError('Invalid arguments to constructor')

    def copy(self):
        return self.__class__(*[copy2(getattr(self, f)) for f in self.fields])

    def allclose(self, other):
        return all([np.allclose(getattr(self, f), getattr(other, f)) for f in self.fields])

    def set_mutable(self, val):
        self.mutable = val
        for f in self.fields:
            a = getattr(self, f)
            if isinstance(a, np.ndarray):
                a.flags.writeable = val



class ArrayStruct(Struct):
    operations = ['add', 'sub', 'mul', 'smul', 'div', 'sdiv']
    pos_fields = []
    recursive_fields = set()
    mutable = True
    match_shapes = False
    check_op_class = True
    require_garrays = False

    def __init__(self, *args, **kwargs):
        Struct.__init__(self, *args, **kwargs)
        if self.match_shapes:
            for a in args:
                if np.shape(a) == np.shape(args[0]):
                    raise RuntimeError('Shapes must match')

        if self.require_garrays:
            self._check_garrays()

    def _check_garrays(self):
        for f in self.fields:
            a = getattr(self, f)
            assert not isinstance(a, np.ndarray)
            assert not isinstance(a, np.float64)

        for f in self.recursive_fields:
            getattr(self, f)._check_garrays()

    def convert_to_garrays(self):
        result = []
        for f in self.fields:
            if f in self.recursive_fields:
                result.append(getattr(self, f).convert_to_garrays())
            else:
                result.append(gnp.garray(getattr(self, f)))
        return self.__class__(*result)

    def _check_op(self, op):
        if op not in self.operations:
            raise RuntimeError('%s: %s not allowed' % (self.__class__, op))

    def _check_same(self, other, op):
        if self.check_op_class and not isinstance(other, self.__class__):
            raise RuntimeError('%s: %s not defined for type %s' % (self.__class__, op, other.__class__))

    def _check_mutable(self):
        if not self.mutable:
            raise RuntimeError('%s is immutable' % self.__class__)

    def __neg__(self):
        self._check_op('smul')
        if len(self.pos_fields) > 0:
            raise RuntimeError('%s: negation not allowed' % self.__class__)
        return self.__class__(*[-getattr(self, f) for f in self.fields])

    def __add__(self, other):
        self._check_op('add')
        self._check_same(other, 'add')
        return self.__class__(*[getattr(self, f) + getattr(other, f) for f in self.fields])


    def __iadd__(self, other):
        self._check_op('add')
        self._check_same(other, 'add')
        self._check_mutable()
        for f in self.fields:
            a = getattr(self, f)
            a += getattr(other, f)
            setattr(self, f, a)
        return self

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        self._check_op('sub')
        self._check_same(other, 'sub')
        return self.__class__(*[getattr(self, f) - getattr(other, f) for f in self.fields])

    def __isub__(self, other):
        self._check_op('sub')
        self._check_same(other, 'sub')
        self._check_mutable()
        for f in self.fields:
            a = getattr(self, f)
            a -= getattr(other, f)
            setattr(self, f, a)
        return self

    def __rsub__(self, other):
        self._check_op('sub')
        self._check_same(other, 'sub')
        return self.__class__(*[getattr(other, f) - getattr(self, f) for f in self.fields])
    
    def __mul__(self, other):
        if np.isscalar(other):
            self._check_op('smul')
            if self.require_garrays and isinstance(other, np.float64):
                other = float(other)
            return self.__class__(*[getattr(self, f) * other for f in self.fields])
        self._check_op('mul')
        self._check_same(other, 'mul')
        return self.__class__(*[getattr(self, f) * getattr(other, f) for f in self.fields])

    def __imul__(self, other):
        self._check_mutable()
        if np.isscalar(other):
            self._check_op('smul')
            if self.require_garrays and isinstance(other, np.float64):
                other = float(other)
            for f in self.fields:
                a = getattr(self, f)
                a *= other
                setattr(self, f, a)
            return self
        self._check_op('mul')
        self._check_same(other, 'mul')
        for f in self.fields:
            a = getattr(self, f)
            a *= getattr(other, f)
            setattr(self, f, a)
        return self

    def __rmul__(self, other):
        return self * other

    def __div__(self, other):
        if np.isscalar(other):
            self._check_op('sdiv')
            if self.require_garrays and isinstance(other, np.float64):
                other = float(other)
            return self.__class__(*[getattr(self, f) / other for f in self.fields])
        self._check_op('div')
        self._check_same(other, 'div')
        return self.__class__(*[getattr(self, f) / getattr(other, f) for f in self.fields])

    def __idiv__(self, other):
        self._check_mutable()
        if np.isscalar(other):
            self._check_op('sdiv')
            if self.require_garrays and isinstance(other, np.float64):
                other = float(other)
            for f in self.fields:
                a = getattr(self, f)
                a /= other
                setattr(self, f, a)
            return self
        self._check_op('div')
        self._check_same(other, 'div')
        for f in self.fields:
            a = getattr(self, f)
            a /= getattr(other, f)
            setattr(self, f, a)
        return self

    def __rdiv__(self, other):
        if np.isscalar(other):
            self._check_op('sdiv')
            if self.require_garrays and isinstance(other, np.float64):
                other = float(other)
            return self.__class__(*[other / getattr(self, f) for f in self.fields])
        self._check_op('div')
        self._check_same(other, 'div')
        return self.__class__(*[getattr(other, f) / getattr(self, f) for f in self.fields])

    def perturb(self, eps):
        if self.recursive_fields:
            raise NotImplementedError()
        
        result = []
        for f in self.fields:
            a = copy2(getattr(self, f))
            if f in self.pos_fields:
                a *= np.exp(np.random.normal(0., eps, size=np.shape(a)))
            else:
                a += np.random.normal(0., eps, size=np.shape(a))
            result.append(a)
        return self.__class__(*result)

    def __len__(self):
        return len(getattr(self, self.fields[0]))

    def __slice__(self, slc):
        return self.__class__(*[getattr(self, f)[slc] for f in self.fields])
        
    def __getitem__(self, slc):
        return self.__class__(*[getattr(self, f)[slc] for f in self.fields])

    def __setitem__(self, slc, other):
        self._check_mutable()
        for f in self.fields:
            getattr(self, f)[slc] = getattr(other, f)

    def __repr__(self):
        args = ', '.join(['%s=%r' % (f, getattr(self, f)) for f in self.fields])
        return '%s(%s)' % (self.__class__, args)

    @classmethod
    def simple_random(cls, size):
        """Generate a random instance for use in testing"""
        if cls.recursive_fields:
            raise NotImplementedError()
        
        result = []
        for f in cls.fields:
            if f in cls.pos_fields:
                result.append(np.random.uniform(0.5, 2., size=size))
            else:
                result.append(np.random.normal(size=size))
        return cls(*result)



class Type:
    pass

class Choice(Type):
    def __init__(self, choices):
        self.choices = choices
        
    def check_instance(self, v):
        return any([check_instance(v, c) for c in self.choices])

    def __repr__(self):
        return 'Choices(' + ', '.join(map(str, self.choices)) + ')'

class Tuple(Type):
    def __init__(self, parts):
        self.parts = parts

    def check_instance(self, v):
        return isinstance(v, tuple) and \
               len(v) == len(self.parts) and \
               all([check_instance(vi, pi) for vi, pi in zip(v, self.parts)])

    def __repr__(self):
        return 'Tuple(' + ', '.join(map(str, self.parts)) + ')'

class List(Type):
    def __init__(self, tp):
        self.tp = tp

    def check_instance(self, v):
        if type(v) != list:
            return False

        return all([check_instance(vi, self.tp) for vi in v])
    

def check_instance(v, t):
    if isinstance(t, Type):
        return t.check_instance(v)

    if hasattr(v, '__class__'):
        return True    # for now, don't check classes because of reloading problems

    # if t is a type or class, check of v is an instance; otherwise, check if they are equal
    try:
        return isinstance(v, t)
    except:
        return v == t



class Params:
    class Fields:
        pass
    
    class Defaults:
        pass

    def __setattr__(self, arg, val):
        tp = getattr(self.Fields, arg)
        assert check_instance(val, tp), RuntimeError('Value {} assigned to {}.{} does not satisfy {}'.format(
            val, self.__class__.__name__, arg, tp))
        self.__dict__[arg] = val

    def __init__(self, **kwargs):
        # make sure every field is covered, and no extra fields are passed in
        fields = [k for k in self.Fields.__dict__.keys() if k[0] != '_']
        assert set(kwargs.keys()) == set(fields)

        for arg, val in kwargs.items():
            setattr(self, arg, val)

    @classmethod
    def defaults(klass, name):
        return klass(**copy.deepcopy(getattr(klass.Defaults, name)))

