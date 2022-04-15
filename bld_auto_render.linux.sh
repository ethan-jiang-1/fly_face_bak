#!/bin/bash

echo 'generate posters in linux env'
#ENV_AR_RENDER_ENGINE="CYCLES"
#export ENV_AR_RENDER_ENGINE

ENV_AR_MAX_VARIATION="8"
export ENV_AR_MAX_VARIATION

ENV_AR_COMBINATION_ID="-1"
export ENV_AR_COMBINATION_ID

~/blender/blender -b ./bld_gen/bldmc_render_m00/render_m00.blend -P ./bld_main_auto_render.py