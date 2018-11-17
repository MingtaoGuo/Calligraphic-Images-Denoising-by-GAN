from train import Denoise
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--img_h", type=int, default=256)
    parser.add_argument("--img_w", type=int, default=256)
    parser.add_argument("--img_c", type=int, default=1)
    parser.add_argument("--lambd", type=int, default=100)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--clean_path", type=str, default="./TrainingSet/clean/")
    parser.add_argument("--noised_path", type=str, default="./TrainingSet/noised/")
    parser.add_argument("--save_path", type=str, default="./save_para/")
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.)
    parser.add_argument("--beta2", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=1e-10)

    parser.add_argument("--is_trained", type=bool, default=True)
    parser.add_argument("--testing_path", type=str, default="./TestingData/1.jpg")

    args = parser.parse_args()
    denoise = Denoise(batch_size=args.batch_size, img_h=args.img_h, img_w=args.img_w, img_c=args.img_c, lambd=args.lambd,
                      epoch=args.epoch, clean_path=args.clean_path, noised_path=args.noised_path, save_path=args.save_path,
                      learning_rate=args.learning_rate, beta1=args.beta1, beta2=args.beta2, epsilon=args.epsilon)
    if args.is_trained:
        denoise.test(args.testing_path, args.save_path)
    else:
        denoise.train()