function img_AddNoise = random_noise(img)
% field = ones(256, 256)*255;
% imshow(field);
[r, c] = size(img);
img_AddNoise = img;
R = unidrnd(3);
P_noise_x = randint(1, 1, [R, c-R]);
P_noise_y = randint(1, 1, [R, r-R]);
for i = 1:r
    for j = 1:c
        if sqrt((P_noise_x - i)^2 + (P_noise_y - j)^2) < R
            if 5 - unidrnd(5) >= 1
                img_AddNoise(i, j) = 0;
            end
        end
    end
end
R = unidrnd(6);
P_noise_x = randint(1, 1, [R, c-R]);P_noise_y = randint(1, 1, [R, r-R]);
for i = 1:r
    for j = 1:c
        if j > P_noise_x && j < P_noise_x + R && i > P_noise_y && i < P_noise_y + R
            if 5 - unidrnd(5) >= 1
                img_AddNoise(i, j) = 0;
            end
        end
    end
end
% figure, imshow(field)