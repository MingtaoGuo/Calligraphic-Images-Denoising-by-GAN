img_size = 512;
count = 0;
for j = 1:9
    img = rgb2gray(imread(['C:\Users\gmt\Desktop\Ώ¬Με_IMG\Ώ¬Με_IMG\Ώ¬Με_IMG\', num2str(j), '.jpg']));
    [r, c] = size(img);
    for i = 1:3
        interval_x0 = unidrnd(c - img_size);
        interval_y0 = unidrnd(r - img_size);
        imwrite(img(interval_y0:interval_y0+img_size, interval_x0:interval_x0+img_size), ['C:\Users\gmt\Desktop\Denoise\Dataset\TestDataSet\',num2str(count),'.jpg']);
        count = count + 1;
    end
end