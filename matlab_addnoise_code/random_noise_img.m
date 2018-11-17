clear;
file = dir('C:\Users\gmt\Desktop\Denoise\Dataset\TestDataSet\*.jpg');
for j = 1:size(file, 1)
    img = imread(['C:\Users\gmt\Desktop\Denoise\Dataset\TestDataSet\', file(j).name]);
    bw_img = im2bw(img);
    noise_img = bw_img;
    tic;
    for i = 1:800
        noise_img = random_noise(noise_img);
    end
    imwrite(noise_img, ['C:\Users\gmt\Desktop\Denoise\Dataset\TestDataSet\groundtruth\', file(j).name])
    toc;
end
imshow(noise_img, []);