#!/bin/zsh
# Side-by-side comparison of a capture crop with a reference-frame crop at the same pixel scale.
#   tools/compare/sbs.zsh <ours.png> <WxH+X+Y> <ref frame time, e.g. 0.900> <WxH+X+Y> <out.png>
set -e
ours=$1 og=$2 rt=$3 rg=$4 out=$5
root=${0:A:h:h:h}
ref=$root/artifacts/reference/design-study/frame_$rt.png
magick \( "$ours" -crop "$og" +repage \) \( "$ref" -crop "$rg" +repage \) -background black +append "$out"
echo "$out"
