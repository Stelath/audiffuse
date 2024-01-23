#!/bin/bash

echo "Starting"

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions

# >>> mamba initialize >>>
# !! Contents within this block are managed by 'mamba init' !!
export MAMBA_EXE="/home/korte/.local/bin/micromamba";
export MAMBA_ROOT_PREFIX="/home/korte/micromamba";
__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    if [ -f "/home/korte/micromamba/etc/profile.d/micromamba.sh" ]; then
        . "/home/korte/micromamba/etc/profile.d/micromamba.sh"
    else
        export  PATH="/home/korte/micromamba/bin:$PATH"  # extra space after export prevents interference from conda init
    fi
fi
unset __mamba_setup
# <<< mamba initialize <<<

micromamba activate ml

cd /home/korte/projects/audifuse/
python main.py fit --config configs/audiffuse_default.yaml 