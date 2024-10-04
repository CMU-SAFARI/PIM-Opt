import os
import sys
import numpy as np

## Code adapted from http://www.deepfeatures.org/download.html
## Code adapted from https://github.com/DS3Lab/LambdaML


def parse_tags(tags, tag):
    tag_splits = tags.split(",")
    value_tag = 0
    if len(tag_splits) == 1:
        tag_pair = tag_splits[0].split(":")
        tag_name = tag_pair[0]
        if tag_name == tag:
            if len(tag_pair) == 2:
                value_tag = float(tag_pair[1])
            else:
                value_tag = None
    else:
        for tag_pair in tag_splits:
            tag_name = tag_pair.split(":")[0]
            if tag_name == tag:
                if len(tag_pair) == 2:
                    value_tag = float(tag_pair[1])
                else:
                    value_tag = None

    return value_tag




def preprocess_yfcc100m(data_file_path, out_file_tag_0, out_file_tag_1, tag_0, tag_1):
    count_tag_0 = 0
    count_tag_1 = 0

    data_file = open(data_file_path,'r')

    while (True):
        line = data_file.readline()
        if not line:
            break
        splits = line.split()
        if len(splits) == 4097:
            tag_0_bool = False 
            tag_1_bool = False
            if tag_0 in splits[0]:
                if tag_1 in splits[0]:
                    value_tag_0 = parse_tags(splits[0], tag_0)
                    value_tag_1 = parse_tags(splits[0], tag_1)

                    if value_tag_0 == None or value_tag_1 == None:
                        continue
                    else:
                        if value_tag_0 > value_tag_1:
                            count_tag_0 += 1
                            tag_0_bool = True
                        else:
                            count_tag_1 += 1
                            tag_1_bool = True
                else:
                    count_tag_0 += 1
                    tag_0_bool = True
            else:
                if tag_1 in splits[0]: 
                    count_tag_1 += 1
                    tag_1_bool = True
            splits_string = [str(float(x)) for x in splits[1:4097]]
            if tag_0_bool:
                out_str = str(0) + " " + " ".join(x for x in splits_string)
                out_file_tag_0.write(out_str)
                out_file_tag_0.write('\n')
            elif tag_1_bool:
                out_str = str(1) + " " + " ".join(x for x in splits_string)
                out_file_tag_1.write(out_str)
                out_file_tag_1.write('\n')

    data_file.close()
    return count_tag_0, count_tag_1


if __name__ == "__main__":
    job_id = int(sys.argv[1])
    source_path_prefix = str(sys.argv[2])
    dest_path_prefix_tmp = str(sys.argv[3])
    print("Start job {}".format(job_id))
    dest_path_prefix = dest_path_prefix_tmp + "preprocessing/YFCC100M-HNfc6/initial_preprocessing/"
    tag_0 = 'indoor'
    tag_1 = 'outdoor'

    out_file_path_tag_0 = dest_path_prefix + "YFCC100M_hybridCNN_gmean_fc6_{}_tag_indoor".format(job_id)

    out_file_path_tag_1 = dest_path_prefix + "YFCC100M_hybridCNN_gmean_fc6_{}_tag_outdoor".format(job_id)


    out_file_tag_0 = open(out_file_path_tag_0,'w+')
    out_file_tag_1 = open(out_file_path_tag_1,'w+')

    all_count_tag_0 = 0
    all_count_tag_1 = 0

    for i in [job_id]:
        print("Processing file {}".format(i), flush=True)
        data_file_path = source_path_prefix + "YFCC100M_hybridCNN_gmean_fc6_{}_tag".format(i)
        count_tag_0, count_tag_1 = preprocess_yfcc100m(data_file_path, out_file_tag_0, out_file_tag_1, tag_0, tag_1)

        all_count_tag_0 += count_tag_0
        all_count_tag_1 += count_tag_1


    

    out_file_tag_0.close()
    out_file_tag_1.close()
    print("__job_id={},count_tag_0={}__".format(job_id,all_count_tag_0))
    print("__job_id={},count_tag_1={}__".format(job_id,all_count_tag_1))
    print("Done job {}".format(job_id))

