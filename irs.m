function result = irs(input, lambda, iteration)
%solve 1/2||U-input||_2^2 + lambda/2*||grad U||_1 
%implicit residual solver, implemented by Yuanhao Gong
%please cite the following work if you use this code
%{ 
@ARTICLE{irsnet,
  author={Gong, Yuanhao},
  journal={IEEE Access}, 
  title={IRSnet: An Implicit Residual Solver and Its Unfolding Neural Network With 0.003M Parameters for Total Variation Models}, 
  year={2025},
  volume={13},
  number={},
  pages={10289-10298},
  doi={10.1109/ACCESS.2025.3528637}}
%}
theta = 0.24; 
threshold = lambda/theta;
im = single(input);
bx = zeros(size(im),'single'); 
by = bx;
gx=[zeros([size(im,1),1],'single'),diff(im,1,2)];%gradient
gy=[zeros([1,size(im,2)],'single');diff(im,1,1)];
for it=1:iteration
    [tempx,tempy] = myGradDiv(bx,by,theta); 
    bx = max(min(gx+tempx,threshold),-threshold);%cut
    by = max(min(gy+tempy,threshold),-threshold);%cut    
end
residual=[diff(bx,1,2), -bx(:,end)]+[diff(by,1,1); -by(end,:)];%div
result = im + theta*residual;

%% gradient of divergence of (bx, by)
function [tx,ty]=myGradDiv(bx,by,theta)
kx=[1,-2,1]*theta+[0,1,0];
kxy=[0,1,-1;0,-1,1;0,0,0;]*theta; 
tx = conv2(bx,kx,'same')+conv2(by,kxy,'same');
ty = conv2(bx,kxy','same')+conv2(by,kx','same');
tx(:,1)=bx(:,1);%boundary condition
ty(1,:)=by(1,:);
