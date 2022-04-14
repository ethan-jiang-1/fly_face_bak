#!/bin/bash

echo 'generate posters in macos env'
ENV_AR_MAX_VARIATION="8"
export ENV_AR_MAX_VARIATION
ENV_AR_COMBINATION_ID="-1"
export ENV_AR_COMBINATION_ID

/Applications/Blender.app/Contents/MacOS/Blender -b ./bld_gen/bldmc_render_m00/render_m00.blend  -P ./bld_main_auto_render.py