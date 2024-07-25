import os
cids = os.listdir("dev")
print(cids)
for cid in cids:
    for videoid in os.listdir("dev/{}".format(cid)):
        for video in os.listdir("dev/{}/{}".format(cid,videoid)):
            try:
                os.rename("dev/{}/{}/{}".format(cid,videoid,video),"videos/{}/{}".format(videoid,video))
            except Exception as e:
                print(e)

