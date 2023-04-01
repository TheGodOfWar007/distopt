function proj = projectPoint(f, k1, k2, point_in_W, T_W_C)
% Transform point into camera frame
point_in_C = T_W_C(1:3,1:3)' * (point_in_W - T_W_C(1:3,end));
p = point_in_C(1:2) / point_in_C(3);
rad = norm(p);
proj = f * (1.0 + k1 * rad * rad + k2 * rad * rad * rad * rad) * p;
end