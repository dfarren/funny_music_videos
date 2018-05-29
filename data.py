# -*- coding: utf-8 -*-

import os
import dill
from datetime import datetime, timedelta
import params

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# The CLIENT_SECRETS_FILE variable specifies the name of a file that contains
# the OAuth 2.0 information for this application, including its client_id and
# client_secret.
# CLIENT_SECRETS_FILE = "client_secret.json"

# This OAuth 2.0 access scope allows for full read/write access to the
# authenticated user's account and requires requests to use an SSL connection.
SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']
API_SERVICE_NAME = 'youtube'
API_VERSION = 'v3'


def get_authenticated_service():
    with open('google_api_key', 'r') as f:
        key = f.read()
    return build(API_SERVICE_NAME, API_VERSION, developerKey=key)


# Build a resource based on a list of properties given as key-value pairs.
# Leave properties with empty values out of the inserted resource.
def build_resource(properties):
    resource = {}
    for p in properties:
        # Given a key like "snippet.title", split into "snippet" and "title", where
        # "snippet" will be an object and "title" will be a property in that object.
        prop_array = p.split('.')
        ref = resource
        for pa in range(0, len(prop_array)):
            is_array = False
            key = prop_array[pa]

            # For properties that have array values, convert a name like
            # "snippet.tags[]" to snippet.tags, and set a flag to handle
            # the value as an array.
            if key[-2:] == '[]':
                key = key[0:len(key) - 2:]
                is_array = True

            if pa == (len(prop_array) - 1):
                # Leave properties without values out of inserted resource.
                if properties[p]:
                    if is_array:
                        ref[key] = properties[p].split(',')
                    else:
                        ref[key] = properties[p]
            elif key not in ref:
                # For example, the property is "snippet.title", but the resource does
                # not yet have a "snippet" object. Create the snippet object here.
                # Setting "ref = ref[key]" means that in the next time through the
                # "for pa in range ..." loop, we will be setting a property in the
                # resource's "snippet" object.
                ref[key] = {}
                ref = ref[key]
            else:
                # For example, the property is "snippet.description", and the resource
                # already has a "snippet" object.
                ref = ref[key]
    return resource


# Remove keyword arguments that are not set
def remove_empty_kwargs(**kwargs):
    good_kwargs = {}
    if kwargs is not None:
        for key, value in kwargs.iteritems():
            if value:
                good_kwargs[key] = value
    return good_kwargs


def fetch(client_api, **kwargs):
  kwargs = remove_empty_kwargs(**kwargs)

  response = client_api().list(
    **kwargs
  ).execute()

  return response


def fetch_comments(client, video_id, max_comments_per_video):

    j = 0
    comments_page_token = ''
    comments_for_this_video = {}

    while j < max_comments_per_video:
        # Fetch comments
        try:
            comments = fetch(client.commentThreads,
                             pageToken=comments_page_token,
                             part='snippet,replies',
                             maxResults=min(max_comments_per_video, 100),
                             videoId=video_id,
                             order='relevance',
                             textFormat='plainText'
                             )
        # if comments are disabled
        except HttpError:
            break

        for comment in comments['items']:

            # there shouldn't be collisions because id is a unique key
            comments_for_this_video[comment['snippet']['topLevelComment']['id']] = {
                'text': comment['snippet']['topLevelComment']['snippet']['textDisplay'],
                'likes': comment['snippet']['topLevelComment']['snippet']['likeCount']
            }

            if 'replies' in comment:
                replies = {}
                for reply in comment['replies']['comments']:
                    replies[reply['id']] = {
                        'text': reply['snippet']['textDisplay'],
                        'likes': reply['snippet']['likeCount']
                    }
                # add replies to comments dict
                comments_for_this_video[comment['snippet']['topLevelComment']['id']]['replies'] = replies
            j += 1

        if 'nextPageToken' in comments:
            comments_page_token = comments['nextPageToken']
        else:
            break

    return comments_for_this_video


def add_tags_stats_comments(i, collisions_counter, videos_string, client, max_videos, category_exclude, data,
                            max_comments_per_video, min_comments_per_video):

    videos = fetch(client.videos,
                   maxResults=min(max_videos, 50),
                   part='snippet,statistics',
                   id=videos_string,
                   regionCode='US',
                   )

    for video in videos['items']:

        if category_exclude == video['snippet']['categoryId'] or video['snippet']['channelTitle'] in params.excluded_channels:
            continue

        video_id = video['id']

        # there may be collisions because when querying by date we will likely get
        # some videos we got in the first query.
        if video_id in data:
            collisions_counter += 1
            if collisions_counter % 10 == 0:
                print("%d collisions so far" % collisions_counter)
            continue

        comments_for_this_video = fetch_comments(client, video_id, max_comments_per_video)

        # if this video has too few comments, we ignore it
        if len(comments_for_this_video) < min_comments_per_video:
            print("Ignoring videoId %s because comments are disabled or too few comments" % video_id)
            continue

        data[video_id] = {
            'title': video['snippet']['title'],
            'description': video['snippet']['description'],
            'comments': comments_for_this_video,
            'statistics': video['statistics'],
        }

        if 'tags' in video['snippet']:
            data[video_id]['tags'] = video['snippet']['tags']

        i += 1
        if i % 10 == 0:
            print("Fetched %d videos so far" % i)

    return i, collisions_counter, data


def fetch_videos(client, max_videos, max_comments_per_video, min_comments_per_video,
               category_id='', category_exclude=None, date_to=None):

    data = {}
    i = 0
    collisions_counter = 0
    videos_page_token = ''
    date_iterator_from = None
    date_iterator_to = None

    while i < max_videos:
        # First call is to bring the most popular videos in the category
        videos = fetch(client.search,
                       type='video',
                       pageToken=videos_page_token,
                       maxResults=min(max_videos, 50),
                       part='snippet',
                       order='viewCount',
                       regionCode='US',
                       relevanceLanguage='EN',
                       videoCategoryId=str(category_id),
                       publishedAfter=date_iterator_from,
                       publishedBefore=date_iterator_to
                       )

        if 'items' not in videos or len(videos['items']) == 0:
            print('No more videos in this call.')
            if date_to is None:
                break

            # not first time here
            if date_iterator_from:
                date_iterator_to = date_iterator_from
                date_iterator_from = (datetime.strptime(date_iterator_from, "%Y-%m-%dT%H:%M:%SZ") -
                                      timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%SZ")

            # first time here
            else:
                date_iterator_to = date_to.strftime("%Y-%m-%dT%H:%M:%SZ")
                date_iterator_from = (date_to - timedelta(days=365)).strftime("%Y-%m-%dT%H:%M:%SZ")

            videos_page_token = ''
            continue

        if 'nextPageToken' in videos:
            videos_page_token = videos['nextPageToken']

        # second call is to add tags and statistics to these videos. I could have done this in one call to videos,
        # but the api for videos.list has a bug -> it doesn't filter by categoryId.
        # This is also useful in order to exclude videos in the case when we want to exclude a category.
        videos_string = ''
        for video in videos['items']:
            videos_string += video['id']['videoId'] + ','
        videos_string = videos_string[:-1]

        i, collisions_counter, data = add_tags_stats_comments(i, collisions_counter, videos_string, client, max_videos,
                                                              category_exclude, data, max_comments_per_video,
                                                              min_comments_per_video)


    return data


def main():

    # When running locally, disable OAuthlib's HTTPs verification. When
    # running in production *do not* leave this option enabled.
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
    client = get_authenticated_service()

    data = {}
    categories_id = []

    for category, category_id in params.categories.iteritems():
        print("\nFetching %s" % category)
        categories_id.append(category_id)
        data[category] = fetch_videos(client, params.max_videos_per_category, params.max_comments_per_video,
                                     params.min_comments_per_video, category_id=category_id
                                     )
        if category == 'music':
            with open('oversample', 'r') as f:
                lines = f.readlines()

            videos_string = ''
            for line in lines:
                key, label = line.split()
                videos_string += key+','
            videos_string = videos_string[:-1]
            print("oversample")
            _, _, data[category] = add_tags_stats_comments(i=0, collisions_counter=0,
                                                                         videos_string=videos_string, client=client,
                                                                         max_videos=params.max_videos_per_category,
                                                                         category_exclude=None,
                                                                         data=data[category],
                                                                         max_comments_per_video=params.max_comments_per_video,
                                                                         min_comments_per_video=params.min_comments_per_video)





    print("\nFetching other")
    data['no_comedy'] = fetch_videos(client, params.max_videos_per_category, params.max_comments_per_video,
                                     params.min_comments_per_video, category_exclude=params.categories['comedy']
                                     )

    with open("videos_data", "wb") as f:
        dill.dump(data, f)


if __name__=='__main__':
    main()
