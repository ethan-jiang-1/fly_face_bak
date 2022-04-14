#!/bin/bash

echo 'generate posters in linux env'
ENV_AR_MAX_VARIATION="8"
export ENV_AR_MAX_VARIATION
ENV_AR_COMBINATION_ID="-1"
export ENV_AR_COMBINATION_ID

~/blender-3.0.1-linux-x64/blender -b ./bld_gen/bldmc_render_m00/render_m00.blend -P ./bld_main_auto_render.py