file = dir('C:\Users\gmt\Desktop\Denoise\Dataset\R_{1-3},R_{1-6},0.8,1200\*.jpg');
for i = 1:size(file, 1)
    name = file(i).name;
    img = imread(['C:\Users\gmt\Desktop\Denoise\Dataset\R_{1-3},R_{1-6},0.8,1200\', name]);
    name(length(name)-3+1:length(name)) = 'png';
    imwrite(img, ['C:\Users\gmt\Desktop\新建文件夹\', name]);
end