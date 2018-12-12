import os


def check_dataset(dataset_path, min_imgs):
    cla_total = 0
    img_total = 0
    cla_useful = 0
    img_useful = 0

    cla_list = os.listdir(dataset_path)
    for cla in cla_list:
        cla_total += 1
        cla_img_list = os.listdir(os.path.join(dataset_path, cla))

        img_total += len(cla_img_list)

        if len(cla_img_list) >= min_imgs:
            cla_useful += 1
            img_useful += len(cla_img_list)

    print('total class %s, total images %s, userful class %s, userful images %s' % (
        cla_total, img_total, cla_useful, img_useful))


if __name__ == '__main__':
    # celebrity_path = '/sdc/devdatas/face/DeepGlint/celebrity'
    # msra_path = '/sdc/devdatas/face/DeepGlint/msra'
    celebrity_path = '/sdc/workdatas/face/DeepGlint/celebrity'
    msra_path = '/sdc/workdatas/face/DeepGlint/msra'

    min_imgs = 5

    print('check celebrity')
    check_dataset(celebrity_path, min_imgs)

    print('check msra')
    check_dataset(msra_path, min_imgs)
