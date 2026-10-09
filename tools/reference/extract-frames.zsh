#!/bin/zsh -i
# Generate inspection artifacts without changing the original reference.
# Run through interactive zsh, as required by this project:
# zsh -ic 'source tools/reference/extract-frames.zsh [input.mov] [output-dir]'

function fabrictear_extract_reference() {
  emulate -L zsh
  setopt pipefail

  local reference_tool_path="${(%):-%x}"
  local reference_project_root="${reference_tool_path:A:h:h:h}"

  if [[ "${1:-}" == "--help" ]]; then
    print -r -- 'Usage: source tools/reference/extract-frames.zsh [input.mov] [output-dir]'
    print -r -- 'Defaults: refs/demo.mov and artifacts/reference/study under the project root.'
    print -r -- 'Requires ffmpeg, ffprobe, and shasum. Refuses to overwrite an existing output directory.'
    return 0
  fi
  if (( $# > 2 )); then
    print -u2 -r -- 'Expected at most an input file and an output directory.'
    return 2
  fi

  local reference_input="${1:-${reference_project_root}/refs/demo.mov}"
  local reference_output="${2:-${reference_project_root}/artifacts/reference/study}"
  reference_input="${reference_input:A}"
  reference_output="${reference_output:A}"

  local reference_command
  for reference_command in ffmpeg ffprobe shasum; do
    if ! command -v "$reference_command" >/dev/null 2>&1; then
      print -u2 -r -- "Missing required command: ${reference_command}"
      return 1
    fi
  done
  if [[ ! -f "$reference_input" ]]; then
    print -u2 -r -- "Reference video not found: ${reference_input}"
    return 1
  fi
  if [[ -e "$reference_output" ]]; then
    print -u2 -r -- "Output already exists; choose a new directory: ${reference_output}"
    return 1
  fi

  mkdir -p -- "$reference_output" || return 1
  ffprobe -v error -show_format -show_streams -of json "$reference_input" \
    > "${reference_output}/video-metadata.json" || return 1
  shasum -a 256 "$reference_input" > "${reference_output}/source.sha256" || return 1
  ffmpeg -version > "${reference_output}/ffmpeg-version.txt" 2>&1 || return 1

  # Select the first decoded frame at or after each requested timestamp.
  # showinfo records its actual PTS; the filename records the requested time.
  # Decode from the start to avoid relying on keyframe seeking semantics.
  local -a reference_times=(0.300 0.900 1.500 2.100 2.700 3.300 3.900 4.500)
  local reference_time
  for reference_time in "${reference_times[@]}"; do
    ffmpeg -hide_banner -nostdin -n -i "$reference_input" \
      -vf "select=gte(t\,${reference_time})*isnan(prev_selected_t),showinfo" \
      -frames:v 1 -fps_mode vfr \
      "${reference_output}/frame_${reference_time}.png" \
      > "${reference_output}/frame_${reference_time}.log" 2>&1 || return 1
    if [[ ! -s "${reference_output}/frame_${reference_time}.png" ]]; then
      print -u2 -r -- "No frame at or after ${reference_time}s; check input duration and decode log."
      return 1
    fi
  done

  ffmpeg -hide_banner -loglevel error -nostdin -n \
    -pattern_type glob -i "${reference_output}/frame_*.png" \
    -vf 'scale=396:-1,tile=4x2' -frames:v 1 \
    "${reference_output}/contact-sheet.png" || return 1

  print -r -- "Reference artifacts: ${reference_output}"
  print -r -- 'Contact sheet order: 0.300, 0.900, 1.500, 2.100, 2.700, 3.300, 3.900, 4.500 seconds.'
  print -r -- 'Filenames are requested times. Actual decoded timestamps are in each frame log.'
}

fabrictear_extract_reference "$@"
