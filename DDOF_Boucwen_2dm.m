function [y,yr,yu] = DDOF_Boucwen_2dm(x,GM,x_meta)
% % lb=[-8  0.1    2e5   5.0e5];
% % ub=[-1  10    6e5   1.5e6];
% % x=lb+(ub-lb).*x;
% 
% lb=[0.0  0.5    2e5   5.0e5];
% ub=[0.5  2.5    6e5   1.5e6];
% x=lb+(ub-lb).*x;
% figure
% hold on
% grid on
%  ylabel('r [KN]','FontSize',16,'Fontname', 'Times New Roman');
% xlabel('u [mm]','FontSize',16,'Fontname', 'Times New Roman');
% set(gca,'FontSize',12,'Fontname', 'Times New Roman');
for i_num=1:size(x,1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%[time,GM] =filtter_groundmotion(x(i_num,1) ,x(i_num,2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [time,GM] = pulse_groundmotion(2.25,0.6);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
t=0:1/1024:30;
%%
k1=x(i_num,1);k2=x(i_num,2);
m1=8e2;m2=9e2;
zeta=0.05;
A_ref=x_meta;

k=[k1+A_ref -A_ref;-A_ref k2+A_ref];
m=[m1 0;0 m2];
omega=sqrt(eig(k*inv(m)));
a0=zeta*2*omega(1)*omega(2)/(omega(1)+omega(2));
a1=zeta*2/(omega(1)+omega(2));
c=a0*m+a1*k;

 dx1=0;dx2=0;
 x1=0;x2=0;
 r=0; u=0; e=0;
 dt=1/1024;yE(1)=0;
 for i=1:length(t)
  [ddx1,ddx2]=Newton_law(GM(i),r,dx1,dx2,x1,x2,c,k1,k2);
  dx1=dx1+ddx1*dt;
  dx2=dx2+ddx2*dt;
  x1=x1+dx1*dt;
  x2=x2+dx2*dt;
  u_new=x2-x1;
  du=(u_new-u)/dt;
  u=u_new;
  yu(i)=u;
  [dr,de] = BW_model(du,r,A_ref);
  r=r+dr*dt;
  e=e+de*dt;
  yr(i)=r;
  yE(i+1)=yE(i)+r*du*dt;
 end

y(i_num,1)=max(abs(yu))*1000;

% plot(yu*1000,yr/1000)
% figure
% plot(yu,yr)
end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ddx1,ddx2] = Newton_law(GM,r,dx1,dx2,x1,x2,c,k1,k2)
% The dynamic responce of 2-DOF 
m1=8e3;m2=9e3;
ddx1=GM+r/m1-k1/m1*x1-(c(1,1)/m1*dx1+c(1,2)/m1*dx2);
ddx2=GM-r/m2-k2/m2*x2-(c(2,1)/m2*dx1+c(2,2)/m2*dx2);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [dr,de] = BW_model(du,r,A_ref)
% Bouc-Wen model with stiffness deterioration
% A_ref=5.5e5;
beta_ref=25;
gamma_ref=12.5;
n_ref=1;
de=r*du;

dr=(A_ref-(beta_ref*sign(du*r)+gamma_ref)*(abs(r))^n_ref)*du;
end


% function [time,GM] = pulse_groundmotion(Vp,Tp)
% %PULSE_GROUNDMOTON 此处显示有关此函数的摘要
% dt=0.01;
% t=0:dt:6;
% gamma=10;
% v=0.1;
% ti=0;
% tmax=ti+0.5*gamma*Tp;
% tf=tmax+0.5*gamma*Tp;
% Dr=Vp.*Tp.*(sin(v+gamma.*pi)-sin(v-gamma.*pi))./(4.*pi*(1-gamma.^2));
% vp=(0.5.*Vp.*cos(2.*pi.*(t-tmax)./Tp+v)-Dr./(gamma.*Tp)).*(1+cos(2.*pi.*(t-tmax)./(gamma.*Tp))).*(ti<=t).*(t<=tf);
% ap=gradient(vp)./gradient(t);
% time=t';
% GM=ap';
% end
