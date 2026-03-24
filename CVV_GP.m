function [Xs_new,error]=CVV_GP(Xs,Ys,x_cand,x_m,x_v)

    main_data_dir = '.\main_tasks\';

    [X_n,~]=size(Xs);
    [X_candn,~]=size(x_cand);
    num=zeros(X_candn,1);
    for sss=1:length(x_cand(:,1))
        Dms=zeros(X_n,1);
        for mm=1:length(Xs(:,1))
        Dms(mm,1)=sqrt(sum(((x_cand(sss,:)-x_m)./x_v-(Xs(mm,:)-x_m)./x_v).^2,2));
        end
    [~,index]=min(Dms);
    num(sss,1)=index;
    end

    S_class=cell(X_n,1);
    for hh=1:X_n
    AAA=num==hh;
    S_class{hh,1}=x_cand(AAA,:);
    end
    num_class=zeros(X_n,1);
    error=zeros(X_n,1);
    Y_selecr=zeros(X_n,1);
    for cc=1:X_n
    Xs_Selec=Xs;
    Ys_Selec=Ys;
    Xs_test=Xs_Selec(cc,:);
    % Ys_test=Ys_Selec(cc,:);
    Xs_Selec(cc,:)=[];
    Ys_Selec(cc,:)=[]; 
    filename = [main_data_dir,'CV_task_', 'main_','date', '.mat'];
    save(filename,"Xs_Selec","Ys_Selec")
    filename = [main_data_dir,'CV_task_', 'test_','date', '.mat'];
    save(filename,"Xs_test")

    system('D:\ProgramData\anaconda3\envs\py391\python.exe predict_loo_n_gp.py');
    filename = [main_data_dir,'CV_test_prediction','.mat'];
    Y_selec1=load (filename);
    Y_selec=Y_selec1.mean;
    Y_selecr(cc,:)=Y_selec;
    error(cc,:)=(abs(Ys(cc,:)-Y_selec));
    num_class(cc,:)=length(S_class{cc}(:,1));
    end

    % RMSE=sqrt(sum(error.^2,1)./(length(Xs(:,1))));
    % for dd=1:length(Ys(1,:))
    % EM1(:,dd)=error(:,dd).*(RMSE(1,dd)./sum(RMSE,2));            
    % end
    % EM=sum(EM1,2)+max(error,[],2);
    % errorn=(error-min(error))./(max(error)-min(error));
    % num_classn=(num_class-min(num_class))./(max(num_class)-min(num_class));
    %  [~,index1]=max(errorn.*num_classn);
     [~,index1]=max(error);
    area=S_class{index1};
%     area_record{nn}=area;
    if isempty(area)
    Dins=zeros(X_candn,1);
    for ss=1:length(x_cand(:,1))
        Dins(ss,1)=sqrt(sum(((x_cand(ss,:)-x_m)./x_v-(Xs(index1,:)-x_m)./x_v).^2,2));
    end
    [~,index4]=min(Dins);
    Xs_new=x_cand(index4,:);
    else
    Ds=zeros(length(area(:,1)),1);
    for ss=1:length(area(:,1))
        Ds(ss)=sqrt(sum(((area(ss,:)-x_m)./x_v-(Xs(index1,:)-x_m)./x_v).^2,2));
    end
   [~,index2]=max(Ds);
    Xs_new=area(index2,:);
    end
end