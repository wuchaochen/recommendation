import os
import sys


def change_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError('{} not found'.format(checkpoint_dir))
    checkpoint_file = checkpoint_dir + '/checkpoint'
    with open(checkpoint_file, 'r') as f:
        lines = f.readlines()

    result = []
    for line in lines:
        index_1 = line.find('"')
        index_2 = line.rfind('/')
        n_line = line[:index_1] + ' "' + checkpoint_dir + line[index_2:]
        result.append(n_line)
    with open(checkpoint_file, 'w') as f:
        for l in result:
            f.write(l)



if __name__ == '__main__':
    checkpoint_dir = sys.argv[1]
    print(checkpoint_dir)
    change_checkpoint(checkpoint_dir)
