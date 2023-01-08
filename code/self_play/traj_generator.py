import argparse
import math

def get_parsers():
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_dis', default=200.0, type=float)
    parser.add_argument('--shape', default="sin")
    parser.add_argument('--A', default=1.0, type=float)
    parser.add_argument('--w', default=2 * math.pi / 200.0, type=float)  #默认200m一个周期
    parser.add_argument('--point_perMeter', default=0.25, type=float, help="每米的点数")
    return parser.parse_args()
    
def main():
    args = get_parsers()

    file = open("./data/ref_traj_1","w")

    if args.shape == "line":
        gap_dis = 1.0/args.point_perMeter # 间隔多少米一个点
        dis = 0
        while dis < args.max_dis:
            file.write(str(dis)+' 0'+'\n')
            dis += gap_dis
    elif args.shape == "sin":
        gap_dis = 1.0/args.point_perMeter # 间隔多少米一个点
        dis = 0
        while dis < args.max_dis:
            file.write(str(dis)+' '+str(args.A * math.sin(args.w * dis))+'\n')
            dis += gap_dis
    file.close()
    print("done.")

if __name__ == "__main__":
    main()