# only required if PyCall is not loaded allready
# pkgid = Base.PkgId(Base.UUID(0x438e738f_606a_5dbb_bf0a_cddfbfd45ab0), "PyCall")
# PC = Base.require(pkgid)

PC = PyCall;

PC.py"""
class SymStr():
    def __init__(self, *args, **kwargs):
        self.s = str(*args, **kwargs)
    def __str__(self):
        return self.s.__str__()
    def __repr__(self):
        return f'SymStr("{self.__str__()}")'
"""

sym_str_py_type = PC.py"SymStr";

PC.PyObject( s :: Symbol ) = PC.py"SymStr('$(string(s))')"o
function PC.convert( ::Type{Symbol}, po :: PC.PyObject ) 
    sym_str = PC.pyisinstance( po, sym_str_py_type ) ? po.s : po;
    return Symbol(PC.convert(AbstractString, sym_str))
end
PC.pytype_mapping(sym_str_py_type, Symbol);
nothing

