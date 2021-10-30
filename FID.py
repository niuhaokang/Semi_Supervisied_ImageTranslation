from cleanfid import fid

if __name__ == '__main__':
    dir1 = '/data/haokang/Semi-Supervisied-DataSet/sketch/supervisied/A'
    dir2 = '/data/haokang/Semi-Supervisied-DataSet/sketch/supervisied/B'

    score = fid.compute_fid(dir1, dir2)
    print(score)
