import webbrowser
import os
import pdb


def html_video_lines(videos):
    out = []
    head = '<div class="video-wrap"><div class="video-box"><iframe width="100%" height="100%" src="https://www.youtube.com/embed/'
    mid = '" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe></div><span class="video-title">'
    tail = '</span></div>\n'
    for key in videos:
        title = ""
        out.append(head+key+mid+title+tail)
    return out


def add_videos(videos_html):
    # with is like your try .. finally block in this case
    with open('web/ranking.html', 'r') as f:
        data = f.readlines()

    i = 0
    for line in data:
        if line.strip() == '<div class="video-container">':
            break
        i += 1

    new_data = data[:i+1] + html_video_lines(videos_html) + ['</div>\n', '</body>\n', '</html>\n']
    with open('web/ranking.html', 'w') as f:
        f.writelines(new_data)

    open_in_browser()


def reset():
    videos_html = []
    for i in range(10):
        videos_html.append("-Q2sSul5AsE")

    add_videos(videos_html)


def open_in_browser():
    cwd = os.getcwd()
    webbrowser.open('file://'+cwd+'/web/ranking.html', new=1)
