%% Original DATA to plot Figure 7 in Norman et al. 2017:
% https://www.nature.com/articles/s41467-017-01184-1

clear all
close all

% SET THE PATH:
indir='/media/itzik/data1/data/PHD/forRafi/figdata_for_Rafi';
outdir='/media/itzik/data1/data/PHD/forRafi/figdata_for_Rafi/figures';
if ~exist(outdir,'dir')
    mkdir(outdir);
    disp('Creating Output Directory...')
end

load(fullfile(indir,'figdata.mat'))


COLOR = struct;
COLOR.orange=[0.9608    0.7451    0.1451];
COLOR.red=[0.9490    0.2196    0.0784];
COLOR.blue=[0.0706    0.2    0.6];
COLOR.black=[0    0    0];

figure('color','w','name','raw HFB spectrum'); hold on;
C = [COLOR.red; COLOR.blue; COLOR.orange;];
h = [];
for i = 1:size(figdata.rawSpectrum.X,1)
    h(end+1)=semilogx(figdata.rawSpectrum.X(i,:),figdata.rawSpectrum.Y(i,:),'color',C(i,:),'LineWidth',1,'LineSmoothing','on');

end
axis square
set(gca,'xscale','log')
set(gca,'xtick',[0.01,0.1,0.25,1,10,20,100,200])
labels={0.01,0.1,0.25,1,10,20,100,200};
set(gca,'xticklabel',labels)
xlim([0 12])
xlabel('Frequency (Hz)','fontsize',12)
ylabel('Log power (dB)','fontsize',12)
title({'Ultra-slow HFB fluctuations';'Category selective electrodes'},'FontWeight','normal','FontSize',12);
legend(h,{'Recall','Resting State','Viewing Pictures'},'fontsize',10,'Location','northeast','LineWidth',2);
legend boxoff
saveas(gcf,fullfile(outdir,get(gcf,'name')),'png')
  
figure('color','w','name','norm HFB spectrum'); hold on;
C = COLOR.red;
h1 = shadedErrorBar(figdata.normSpectrum.X,figdata.normSpectrum.Y,figdata.normSpectrum.SD,{'color',C,'LineWidth',1,'LineSmoothing','on'},0.5);
h1 = h1.mainLine;
axis square tight
h2 = plot(get(gca,'xlim'),[0 0],'k-','linewidth',0.5);
set(gca,'xtick',[-2,-1,log10(0.25),0,1,log10(20),2,log10(200)])
labels={0.01,0.1,0.25,1,10,20,100,200};
set(gca,'xticklabel',labels)
xlabel('Frequency (Hz)','fontsize',12)
ylabel('Power gain relative to rest (dB)','fontsize',12)
title({'Ultra-slow HFB fluctuations';'Category selective electrodes'},'FontWeight','normal','FontSize',12);
legend([h1,h2],{'Recall','Rest baseline'},'fontsize',10,'Location','northeast','LineWidth',2);
legend boxoff
saveas(gcf,fullfile(outdir,get(gcf,'name')),'png')
