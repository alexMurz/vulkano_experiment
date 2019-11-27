
@echo off
setlocal enabledelayedexpansion

set api=%1
IF [%api%] == [] (
    set api=vulkan
)


cd src/data/
call BUILD-SPIRV.bat
cd ../../

cargo run --features !api!
