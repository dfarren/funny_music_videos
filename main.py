import sys
import data
import model
import web

if len(sys.argv) < 2:
    web.open_in_browser()

else:
    nbr_videos = int(sys.argv[1])
    data.main()
    model.main(nbr_videos)