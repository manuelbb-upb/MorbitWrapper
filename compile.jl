
if isempty(ARGS)
    OUTFILENAME = "custom_sysimage.so";
else
    OUTFILENAME = ARGS[1];
end
println(ENV["LD_LIBRARY_PATH"])
println(OUTFILENAME)

# create a temp directory in pwd
compiler_env = tempname()

if isdir(compiler_env)
  rm( compiler_env, recursive = true )
end    

mkdir(compiler_env)

# switch environment to temp environment
const Pkg = Base.require(Base.PkgId(Base.UUID("44cfe95a-1eb2-52ea-b672-e2afdf69b78f"), "Pkg"))

Pkg.activate(compiler_env)
@info "Loading PackageCompiler..."	# globally available via LOAD_PATH
using PackageCompiler

@info "Installing PyCall..."
Pkg.add([
    Pkg.PackageSpec(
        name = "PyCall",
        uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0",
    ),
])

using PyCall

#=
Pkg.add([
   Pkg.PackageSpec(url="/project_files/Morbit")
]);

using Morbit
=#

@info "Compiling system image..."
create_sysimage(
    [:PyCall] ; #, :Morbit];
    sysimage_path = OUTFILENAME
)

@info "System image is created as $(joinpath(pwd(),OUTFILENAME))."
rm( compiler_env, recursive = true , force = true);
