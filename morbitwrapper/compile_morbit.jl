# This file is a modified version of the script at
# https://github.com/JuliaPy/pyjulia/blob/master/src/julia/compile.jl
# from the pyjulia projects, which ships under the MIT license

compiler_env, script, output, morbit_repo_url, morbit_env = ARGS

include("prepare_path.jl")

const Pkg =
    Base.require(Base.PkgId(Base.UUID("44cfe95a-1eb2-52ea-b672-e2afdf69b78f"), "Pkg"))

Pkg.activate(compiler_env)
@info "Loading PackageCompiler..."
using PackageCompiler

Pkg.activate(prepare_path(morbit_env))
# install morbit if need be
if !(haskey( Pkg.dependencies(), Base.UUID("88936782-c8cd-4a0f-b259-ffb12bfd2869") ) )
    Pkg.add(; url = morbit_repo_url )
end

@info "Compiling Morbit system image..."
create_sysimage(
    [:Morbit],
    sysimage_path = output,
    precompile_execution_file = script,
)

@info "Morbit image is created at $output"
