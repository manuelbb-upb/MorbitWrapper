#!/bin/sh

export LD_LIBRARY_PATH=$(julia ${BASH_SOURCE[0]}/patch_openssl_jll.jl):$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH
