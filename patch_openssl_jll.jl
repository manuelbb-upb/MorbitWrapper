using Pkg;
using Pkg.Artifacts;

# make a temporary environment
envpath = tempname()
mkdir(envpath)

Pkg.activate( envpath )

Pkg.add( "OpenSSL_jll" )

hash_val = artifact_hash("OpenSSL", joinpath( dirname( Base.find_package("OpenSSL_jll") ), "..", "Artifacts.toml" ) )
lib_path = joinpath( artifact_path(hash_val), "lib" )

Pkg.activate()
rm( envpath, recursive = true, force = true)

println(lib_path)
