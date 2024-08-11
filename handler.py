
from IPython.display import display
from models import *
from config import *
from src.utils import *

class timetable_optimization:
    
    def __init__(self, args, passenger_flow):
        self.args = args
        self.scale = passenger_flow 
        self.alpha = self.alpha_calculator(self.args.peak_time) 
        self.rv = skewnorm(self.alpha)
        self.peak_time_low = skewnorm.ppf(0.01, self.alpha)
        self.peak_time_high = skewnorm.ppf(0.99, self.alpha)
        self.multivariate_monte_carlo_optimization()
        self.plot_result(self.best_timetable, self.args.fixed_timetable)
        
    def alpha_calculator(self, x):
        self.peak_time = x
        root = optimize.bisect(self.der_skew_pdf, a = -15, b = 15)
        return root
    
    def der_skew_pdf(self, alpha):
        self.peak_time_low = skewnorm.ppf(0.01, alpha)
        self.peak_time_high = skewnorm.ppf(0.99, alpha)
        x = (self.peak_time_low - self.peak_time_high)/(60-self.args.fixed_timetable[-1] )*(self.peak_time-60)+self.peak_time_low
        return (7186705221432913*x*np.exp(-x**2/2)*(special.erf((2**(0.5)*alpha*x)/2)/2-1))/9007199254740992+(7186705221432913*2**(1/2)*alpha*np.exp(-(alpha**2*x**2)/2)*np.exp(-x**2/2))/(18014398509481984*np.pi**(1/2))
    
    def f(self,x):
        x = (self.peak_time_low-self.peak_time_high)/(60-self.args.fixed_timetable[-1] )*(x-60)+self.peak_time_low
        return self.rv.cdf(x)*self.scale
    
    def f_dir(self, x):
        return derivative(self.f, x, dx=1e-6)
    
    def generate_sample_point(self,num_variables, fixed_elements = None):
        sample_point = fixed_elements.copy() if fixed_elements is not None else []
        if fixed_elements is not None:
            remaining_elements = [random.choice(list(np.arange(60, self.args.fixed_timetable[-1] , 0.5))) for val in range(num_variables-len(fixed_elements))]
        else:
            remaining_elements = [random.choice(list(np.arange(60, self.args.fixed_timetable[-1] , 0.5))) for val in range(num_variables)]
        sample_point += remaining_elements 
        sample_point = sorted(sample_point)
        return sample_point

    def objective_function(self, x, num_train):
        obj = 0
        for i in range(num_train):
            if i not in [0, num_train-1]:
                temp_i= x[i]
                temp_i_2 = x[i-1]
                obj += (int((self.f(temp_i)-self.f(temp_i_2))))**3*int((temp_i-temp_i_2))**2 
            elif i == 0:  
                obj += (int((self.f(x[i])-self.f(60))))**3*int((x[i]-60))**2
            elif i == num_train-1:
                obj += (int((self.f(self.args.fixed_timetable[-1] )-self.f(x[i-1]))))**3*int((self.args.fixed_timetable[-1] -x[i-1]))**2
        
        return obj//(num_train+2)
    
    def multivariate_monte_carlo_optimization(self):
        sample_point = self.generate_sample_point(self.args.num_train, self.args.fixed_timetable)
        best_value = self.objective_function(sample_point, self.args.num_train)
        self.best_timetable = sample_point

        for j in range(self.args.num_samples-1):
            sample_point = self.generate_sample_point(self.args.num_train, self.args.fixed_timetable)
            value = self.objective_function(sample_point, self.args.num_train)
            if value < best_value:
                best_value = value
                self.best_timetable = sample_point
            sample_point = [f for f in sample_point if f not in self.args.fixed_timetable]

        best_timetable_pd = pd.DataFrame(np.array(self.best_timetable).reshape(1,-1),
             columns=[f'{str(i+1)}{"st" if i+1 == 1 else "nd" if i+1 == 2 else "th"} train' for i in range(len(self.best_timetable))],
             index=['Departure time of the train'])

        display(best_timetable_pd)
    
    def plot_result(self, best_timetable, fixed_time = None):
        self.end_time = fixed_time[-1]
        x = np.linspace(60,self.end_time,10000)
        fig, ax = plt.subplots()
        fig.set_size_inches((15/2, 5))
        ax.plot(x, self.f(x), 
               color = 'black', lw=5, alpha=0.6, label='passengers')
        
        if fixed_time is not None:
            best_timetable = [best_sol for best_sol in best_timetable if best_sol not in fixed_time]
        for i in best_timetable:
            ax.vlines(x = i, color = 'b',linestyle = '--',ymin = self.f(60), ymax = self.f(i))
            ax.hlines(y = self.f(i), color = 'b', linestyle='--', xmin = 60, xmax = i)
        for j in fixed_time:
            ax.vlines(x = j, color = 'r',linestyle = '--',ymin = self.f(60), ymax = self.f(j))
            ax.hlines(y = self.f(j), color = 'r', linestyle='--', xmin = 60, xmax = j)
        
        ax.set_xlabel('Time(min)')
        ax.set_ylabel('Passengers(number)')
        ax.set_title('Optimized Metro extension timetable')
        plt.savefig('./result/Optimized_Metro_extension_timetable.pdf', pad_inches = 0, bbox_inches='tight', format='pdf')
        plt.show()

def save_model(model, model_dir, epoch=None):
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, 'StemGNN.pt')
    with open(file_name, 'wb') as f:
        torch.save(model, f)


def load_model(model_dir, epoch=None):
    if not model_dir:
        return
    epoch = str(epoch) if epoch else ''
    file_name = os.path.join(model_dir, 'StemGNN.pt')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(file_name):
        return
    with open(file_name, 'rb') as f:
        model = torch.load(f)
    return model


def inference(model, dataloader, device, node_cnt, window_size, horizon):
    forecast_set = []
    target_set = []
    model.eval()
    with torch.no_grad():
        for i, (inputs, target) in enumerate(dataloader):
            inputs = inputs.to(device)
            target = target.to(device)
            step = 0
            forecast_steps = np.zeros([inputs.size()[0], horizon, node_cnt], dtype=np.float64)
            while step < horizon:
                forecast_result, a = model(inputs)
                len_model_output = forecast_result.size()[1] #node_cnt 
                if len_model_output == 0:
                    raise Exception('Get blank inference result')
                inputs[:, :window_size - len_model_output, :] = inputs[:, len_model_output:window_size, :].clone()
                inputs[:, window_size - len_model_output:, :] = forecast_result.clone()
                forecast_steps[:, step:min(horizon - step, len_model_output) + step, :] = \
                    forecast_result[:, :min(horizon - step, len_model_output), :].detach().cpu().numpy()
                
                step += min(horizon - step, len_model_output)
            forecast_set.append(forecast_steps)
            target_set.append(target.detach().cpu().numpy())
    return np.concatenate(forecast_set, axis=0), np.concatenate(target_set, axis=0)

def validate(model, dataloader, device, normalize_method, statistic,
             node_cnt, window_size, horizon,station_num):
                
    forecast_norm, target_norm = inference(model, dataloader, device,
                                           node_cnt, window_size, horizon)
    if normalize_method and statistic:
        forecast = de_normalized(forecast_norm, normalize_method, statistic)
        target = de_normalized(target_norm, normalize_method, statistic)
    else:
        forecast, target = forecast_norm, target_norm
    
    score = evaluate(np.array(target[0][0])[station_num], np.array(forecast[0][0])[station_num])
    score_by_node = evaluate(target, forecast, by_node=True)
    score_norm = evaluate(target_norm, forecast_norm)
    
    return dict(mae=score[1], mae_node=score_by_node[1], mape=score[0], mape_node=score_by_node[0],
                rmse=score[2], rmse_node=score_by_node[2],forecast=forecast, target=target)

def train(data, args,result_file):

    node_cnt = data.shape[1]
    model = Model(node_cnt, 2, args.window_size, args.multi_layer, horizon=args.horizon)
    model.to(args.device)
    if len(data) == 0:
        raise Exception('Cannot organize enough training data')

    if args.norm_method == 'z_score':
        train_mean = np.mean(data, axis=0)
        train_std = np.std(data, axis=0)
        normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
    elif args.norm_method == 'min_max':
        train_min = np.min(data, axis=0)
        train_max = np.max(data, axis=0)
        normalize_statistic = {"min": train_min.tolist(), "max": train_max.tolist()}
    else:
        normalize_statistic = None
    if normalize_statistic is not None:
        with open(os.path.join(result_file, 'norm_stat.json'), 'w') as f:
            json.dump(normalize_statistic, f)
    
    if args.optimizer == 'RMSProp':
        my_optim = torch.optim.RMSprop(params=model.parameters(), lr=args.lr, eps=1e-08)
    else:
        my_optim = torch.optim.Adam(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=args.decay_rate)

    train_set = ForecastDataset(data, window_size=args.window_size, horizon=args.horizon,
                                normalize_method=args.norm_method, norm_statistic=normalize_statistic)
    train_loader = torch_data.DataLoader(train_set, batch_size=args.batch_size, drop_last=False, shuffle=False,
                                         num_workers=0)
    forecast_loss = nn.MSELoss(reduction='mean').to(args.device)

    best_validate_mae = np.inf
    validate_score_non_decrease_count = 0
    performance_metrics = {}

    slicing_valid_sample = list(set([random.randrange(1,len(train_set)-1) for i in range(int(len(train_set)*args.valid_ratio))]))
    slicing_train,slicing_valid,slicing_test = [],[],[] 

    for i,(x,y) in enumerate(train_set) :
        if i in [len(train_set)-1] :
            slicing_test.append((x,y))
        elif i in (slicing_valid_sample) :
            slicing_valid.append((x,y))
        else :
            slicing_train.append((x,y))
            
    train_loader = torch_data.DataLoader(slicing_train, batch_size=args.batch_size, drop_last=False, shuffle=True, num_workers=0)
    valid_loader = torch_data.DataLoader(slicing_valid, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    for epoch in range(args.epoch):
        epoch_start_time = time.time()
        model.train()
        loss_total = 0
        cnt = 0
        for i, (inputs, target) in enumerate(train_loader):
            
            inputs = inputs.to(args.device)
            target = target.to(args.device)
            model.zero_grad()
            forecast, attention_tmp = model(inputs)
            target_sample = target[:,:,args.station_index]
            forecast_sample = forecast[:,:,args.station_index]
            loss = forecast_loss(forecast_sample, target_sample)
            cnt += 1
            loss.backward()
            my_optim.step()
            loss_total += float(loss)
            
        if (epoch + 1) % args.exponential_decay_step == 0:
            my_lr_scheduler.step()
        if (epoch + 1) % args.validate_freq == 0:
            print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f}'.format(epoch+1, (time.time() - epoch_start_time), loss_total / cnt))

        is_best_for_now = False
                
        pm = validate(model, valid_loader, args.device, args.norm_method, normalize_statistic,
                     node_cnt, args.window_size, args.horizon, station_num=args.station_index)
        
        if best_validate_mae > pm['mae']:
            best_mae,best_mape,best_rmse = pm['mae'],pm['mape'],pm['rmse']
            is_best_for_now = True
            validate_score_non_decrease_count = 0
        else:
            validate_score_non_decrease_count += 1
            
        if is_best_for_now:
            save_model(model, result_file, epoch)
        if args.early_stop and validate_score_non_decrease_count >= args.early_stop_step:
            print('Epoch:{}'.format(epoch),end=' | ')
            break
    
    print('\n'+'#'*20)
    print('Best validate performance:')
    print(f'MAE:{round(best_mae, 3)} | MAPE:{round(best_mape, 3)} | RMSE:{round(best_rmse, 3)}')
    print('#'*20+'\n')

    return slicing_test


def test(data, test_set, args, result_test_file):
    
    node_cnt = data.shape[1]
    with open(os.path.join(result_test_file, 'norm_stat.json'),'r') as f:
        normalize_statistic = json.load(f)

    model = load_model(result_test_file)
    test_loader = torch_data.DataLoader(test_set, batch_size=args.batch_size, drop_last=False,
                                        shuffle=False, num_workers=0)
    
    performance_metrics = validate(model, test_loader, args.device, args.norm_method, normalize_statistic,
                  node_cnt, args.window_size, args.horizon, station_num=args.station_index)
    
    return performance_metrics

def stemgnn (args) :

    result_test_file = os.path.join('result', args.dataset)
    if not os.path.exists(result_test_file):
        os.makedirs(result_test_file)
    data_file = os.path.join('dataset', args.dataset + '.csv')
    data = pd.read_csv(data_file)
    data.loc[len(data)] = data.loc[len(data)-1].copy()
    data = data.values

    try:
        inference = train(data, args, result_test_file)
        performance_metrics = test(data, inference, args, result_test_file)
    except KeyboardInterrupt:
        print('\n'+'#'*20)
        print('Keyboard Interrupt')
        print('#'*20+'\n')

    return int(performance_metrics['forecast'].sum()) // 2

def main (args) :
  passenger_flow = stemgnn(args)
  timetable_optimization(args, passenger_flow)
