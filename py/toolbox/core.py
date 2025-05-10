from .version import __version__

def hello(to): 
    print('hello %s; (code version: %s)'%(to, __version__) )

def hello_world():
    '''
    Return
    ------
    out: str
         "hello world"
    '''
    out = "hello world"
    return out
