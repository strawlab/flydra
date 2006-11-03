# NOTE: frame numbers *must* start with 1
ffmpeg -b 8000 -f mpeg2video -r 30 -i frame%06d.png movie.mpeg
