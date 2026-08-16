function rho = mapSOS2Density(sos)
    % mapSOS2Density maps speed of sound (SoS) to density.
    % Input:
    %   sos —— 3D array of size [Nx, Ny, Nz], in m/s
    % Output:
    %   rho —— Array with the same size as sos, in kg/m^3

    [nr, nc, nz] = size(sos);
    rho = zeros(nr, nc, nz, 'single');

    % Speed-of-sound range
    cmin = 1400; 
    cmax = 3700;
    
    % Minimum-density distribution: N(0.91 g/cm^3, 0.01^2)
    mu_min = 0.91 * 1000;       % 910 kg/m^3
    sigma_min = 0.01 * 1000;    % 10 kg/m^3

    % Parameters of four distributions for the maximum density
    % (bone density). Values originally given in g/cm^2 are treated
    % as g/cm^3 and converted to kg/m^3.
    bone_means = [0.991, 1.119, 0.970, 1.079] * 1.8557 * 1000;   % kg/m^3
    bone_stds  = [0.10,  0.13,  0.11,  0.13]  * 1.8557 * 100;   % kg/m^3

    % Generate one minimum and maximum density value for each slice
    normal_min = mu_min + sigma_min * randn(1, nz);

    % Randomly select one of the four bone-density distributions
    % for each slice and draw one sample from the selected distribution
    idx = randi(4, [1, nz]);  % Random distribution index
    normal_max = zeros(1, nz);
    for k = 1:nz
        normal_max(k) = bone_means(idx(k)) + ...
                        bone_stds(idx(k)) * randn();
    end

    % Linear mapping:
    % rho = normal_min + (sos - cmin)/(cmax - cmin) ...
    %       * (normal_max - normal_min)
    %
    % Process one slice at a time to reduce memory usage
    for k = 1:nz
        slice = single(sos(:,:,k));

        % Normalized SoS value
        w = (slice - cmin) / (cmax - cmin);

        % Linearly map SoS to [normal_min(k), normal_max(k)]
        rho(:,:,k) = normal_min(k) + ...
                     w * (normal_max(k) - normal_min(k));
    end
end