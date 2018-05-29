from datetime import datetime

categories = {
             'comedy': 23,
             'music': 10
}

max_videos_per_category = 1000  # number of videos to fetch, if possible (youtube caps this at around 500 videos)
max_comments_per_video = 50  # number of comments to fetch, if possible
min_comments_per_video = 3  # minimum number of comments a video needs to have in order to fetch it
date_from_filling_videos = datetime(2005,01,01)
date_to_filling_videos = datetime.now()

excluded_channels = {"Jimmy Kimmel Live", "The Independent"}