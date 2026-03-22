clear ; clc; close all;
startup_rvc;
% 机器人各连杆DH参数
d1 = 0;
d2 = 0;
d3 = 0;
% % 由于关节4为移动关节，故d4为变量，theta4为常量
% theta4 = 0;
a1 = 62;
a2 = 93;
a3 = 150;
% a4 = 0;

alpha1 = 90 / 180 * pi;
alpha2 = 0 / 180 * pi;
alpha3 = 0 / 180 * pi;
% alpha4 = 0 / 180 * pi;
% 定义各个连杆，默认为转动关节
%           theta      d        a        alpha 
L(1)=Link([  0         d1      a1      alpha1]); L(1).qlim=[-pi/2,pi/2];
L(2)=Link([  0         d2      a2      alpha2]); L(2).qlim=[-pi/2,pi/2]; %L(2).offset=pi/2;
L(3)=Link([  0         d3      a3      alpha3]); L(3).qlim=[-3*pi/4,3*pi/4];
% 移动关节需要特别指定关节类型--jointtype
% L(4)=Link([theta4       0      a4      alpha4]); L(4).qlim=[0,180]; L(4).jointtype='P';
% 把上述连杆“串起来”
WWW=SerialLink(L,'name','WWW');
% 定义机器人基坐标和工具坐标的变换
WWW.base = transl(0 ,0 ,0);
WWW.tool = transl(0 ,0 ,0);
WWW.teach();

x=[];
y=[];
z=[];

for theta1 = -pi/2:0.2:pi/2
    for theta2 = -pi/2:0.2:pi/2
        for theta3 = -3*pi/4:0.2:3*pi/4
            T=WWW.fkine([theta1,theta2,theta3]);
            x=[x T.t(1,1)];
            y=[y T.t(2,1)];
            z=[z T.t(3,1)];
        end
    end
end
scatter3(x,y,z, '.');
xlabel('x');
ylabel('y');
zlabel('z');
grid on;


