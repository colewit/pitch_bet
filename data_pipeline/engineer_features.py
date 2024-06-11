
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid



import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


pd.options.display.max_columns = 100
pd.set_option('future.no_silent_downcasting', True)

def get_events(df):
    df['first_pitch'] = np.logical_and(df.at_bat_number==1, df.pitch_number==1)
    
    df['day_game_number'] = \
        df.groupby(['home_team','away_team','game_date'], sort=False)['first_pitch'].transform('cumsum')
    
    
    df['walk'] = np.logical_and(df.balls ==3, df.type=='B')
    df['walk'] = np.where(df.events.str.contains('interf'),False, df.walk)
    
    df['strikeout'] = np.logical_and(df.strikes ==2, np.logical_and(df.type=='S', ~(df.events.isna())))
    df['strikeout'] = np.where(df.events.str.contains('interf'),False, df.strikeout)
    
    df['single'] = df.events=='single'
    df['double'] = df.events=='double'
    df['triple'] = df.events=='triple'
    df['homerun'] = df.events.str.contains('home_run').fillna(False)
    
    df['hit'] = (df.single + df.double+df.triple+df.homerun).astype(bool)
    df['runs'] = df.post_bat_score - df.bat_score
    return df


def get_linear_weights(df):
    df['player_at_bat_number'] = df\
        .groupby(['game_date','pitcher','batter','day_game_number'])\
        .at_bat_number.rank('dense')
    
    df = df.sort_values('pitch_number')
    df['row_number'] = range(len(df))

    df['out'] = ~df.events.isna()*(1-df[['walk','single','homerun','triple','double']].sum(axis=1)).astype(bool)
    df['end_of_at_bat'] = df[['walk','single','homerun','triple','double', 'out']].sum(axis=1).astype(bool)


    df['linear_weight'] = np.where(
        df.end_of_at_bat,
        df.walk*.55+df.single*.7+df.double*1+df.triple*1.27+df.homerun*1.65 - .26,
        np.nan)

        
    df['linear_weight_if_at_bat_over'] = np.where(df.end_of_at_bat,
                                            (df.linear_weight + .26).clip(0),
                                             np.nan)

    df['end_of_ab_linear_weight'] = df\
        .groupby(['game_date','batter','pitcher','player_at_bat_number'])\
        .linear_weight_if_at_bat_over.transform('mean')

    df = df[(df.balls<4) & (df.strikes < 3)]
    
    counts = df\
        .groupby(['balls','strikes']).agg(count_value = ('end_of_ab_linear_weight','mean')).reset_index()
    
    counts = counts.sort_values('strikes', ascending=False)
    
    counts['strike_value'] = -1*counts.groupby('balls').count_value.diff()
    
    counts = counts.sort_values('balls', ascending=False)
    
    counts['ball_value'] = -1*counts.groupby('strikes').count_value.diff()
    
    df = df.merge(counts, how = 'left', on=['balls','strikes'])
    
    df['out'] = ~df.events.isna()*(1-df[['walk','single','homerun','triple','double']].sum(axis=1)).astype(bool)
    df['end_of_at_bat'] = df[['walk','single','homerun','triple','double', 'out']].sum(axis=1).astype(bool)

    df['ball'] = df.description.isin(['ball','blocked_ball','intent_ball','pitchout'])
    df['strike'] = ~df.ball
    df['foul'] = df.description.isin(['foul','foul_bunt','foul_pitchout'])
    
    df['foul_with_2_strikes'] = np.logical_and(df.foul, df.strikes==2)
    
    df['linear_weight'] = np.where(df.end_of_at_bat, 
                                   df.linear_weight,
                                   df.strike_value.fillna(0)*df.strike + df.ball_value.fillna(0)*df.ball)
    
    df = df.dropna(subset='linear_weight')
    
    df['estimated_linear_weight'] = df.estimated_woba_using_speedangle/1.2
    
    df['estimated_linear_weight'] = df.estimated_linear_weight.combine_first(df.linear_weight)
    
    return df
    
def pitch_physics(df, drop_intermediates = True):
    
    g_fts = 32.174
    R_ball = .121
    mass = 5.125
    circ = 9.125
    temp = 72
    humidity = 50
    pressure = 29.92
    temp_c = (5/9)*(temp-32)
    pressure_mm = (pressure * 1000) / 39.37
    svp = 4.5841 * np.exp((18.687 - temp_c/234.5) * temp_c/(257.14 + temp_c))
    rho = (1.2929 * (273 / (temp_c + 273)) * (pressure_mm - .3783 *
                                              humidity * svp / 100) / 760) * .06261
    const = 0.07182 * rho * (5.125 / mass) * (circ / 9.125)**2
    
    df['release_y'] = 60.5-df.release_extension
    df['t_back_to_release'] = (-df.vy0-np.sqrt(df.vy0**2-2*df.ay*(50-df.release_y)))/df.ay
    df['vx_r'] = df.vx0+df.ax*df.t_back_to_release
    df['vy_r'] = df.vy0+df.ay*df.t_back_to_release
    df['vz_r'] = df.vz0+df.az*df.t_back_to_release
    df['t_c'] = (-df.vy_r - np.sqrt(df.vy_r**2 - 2*df.ay*(df.release_y - 17/12))) / df.ay
    df['calc_x_mvt'] = (df.plate_x-df.release_pos_x-(df.vx_r/df.vy_r)*(17/12-df.release_y))
    df['calc_z_mvt'] = (df.plate_z-df.release_pos_z-(df.vz_r/df.vy_r)*(17/12-df.release_y))+0.5*g_fts*df.t_c**2
    
    
    df['vx_bar'] = (2 * df.vx_r + df.ax * df.t_c) / 2
    df['vy_bar'] = (2 * df.vy_r + df.ay * df.t_c) / 2
    df['vz_bar'] = (2 * df.vz_r + df.az * df.t_c) / 2
    df['v_bar'] = np.sqrt(df.vx_bar**2 + df.vy_bar**2 + df.vz_bar**2)
    
    df['adrag'] = -(df.ax * df.vx_bar + df.ay * df.vy_bar + (df.az + g_fts) * df.vz_bar)/df.v_bar
    df['amagx'] = df.ax + df.adrag * df.vx_bar/df.v_bar
    df['amagy'] = df.ay + df.adrag * df.vy_bar/df.v_bar
    df['amagz'] = df.az + df.adrag * df.vz_bar/df.v_bar + g_fts
    df['amag'] = np.sqrt(df.amagx**2 + df.amagy**2 + df.amagz**2)
    
    df['Cd'] = df.adrag / (df.v_bar**2 * const)
    df['Cl'] = df.amag / (df.v_bar**2 * const)
    
    
    df['spin_t'] = 78.92*0.4*df.Cl/(1-2.32*df.Cl)*df.v_bar
    
    df['phi'] = np.where(df.amagz.fillna(-1)>0,
                         np.arctan2(df.amagz, -df.amagx) * 180/np.pi,
                         360+np.arctan2(df.amagz, -df.amagx) * 180/np.pi)
    
    df['phi'] = np.where(df.amagz.isna(), np.nan, df.phi)
    
    df['tilt'] = np.where(3-(1/30)*df.phi.fillna(0)<=0, 3-(1/30)*df.phi + 12, 3-(1/30)*df.phi)
    df['tilt'] = np.where(df.phi.isna(), np.nan, df.tilt)
    
    df['spin_efficiency'] = df.spin_t/df.release_spin_rate
    
    columns_to_drop = ['release_y', 't_back_to_release', 'vx_r', 'vy_r',
           'vz_r', 't_c', 'calc_x_mvt', 'calc_z_mvt', 'vx_bar', 'vy_bar', 'vz_bar',
           'v_bar', 'adrag', 'amagx', 'amagy', 'amagz', 'amag', 'Cd', 'Cl',
           'spin_t', 'phi']
    if drop_intermediates:
        df = df.drop(columns=columns_to_drop)
    return df

def pitcher_specific_metrics(df):
    df['release_speed_max'] = df.groupby('pitcher').release_speed.transform('max')
    df['portion_of_max_velo'] = df.release_speed/df.release_speed_max
    df['velo_percentile_for_pitcher'] = df.groupby('pitcher').release_speed.rank(pct=True)
    df['armside_horz_break'] = np.where(df.p_throws=='L', -1*df.pfx_x, df.pfx_x)
    df['armside_tilt'] = np.where(df.p_throws=='L', 12-df.tilt, df.tilt)
    
    df['fastball'] = df.pitch_type == 'FF'
    df['has_fastball'] = df.groupby('pitcher').fastball.transform('max')
    
    # if you dont have a 4 seam make more inclusive
    df['fastball'] = np.where(df.has_fastball, df.fastball, df.pitch_type.isin(['FF','FC','SI']))
    df['fastball_tilt'] = np.where(df.fastball, df.tilt, np.nan)
    
    df['fastball_tilt'] = df.groupby('pitcher').fastball_tilt.transform('mean')
    df['tilt_off_fb'] = df.tilt - df.fastball_tilt
    
    df['fastball_horz_break'] = df.groupby('pitcher').armside_horz_break.transform('mean')
    df['horz_break_off_fb'] = df.armside_horz_break- df.fastball_horz_break
    
    df['fastball_vert_break'] = df.groupby('pitcher').pfx_z.transform('mean')
    df['vert_break_off_fb'] = df.pfx_z- df.fastball_vert_break
    return df







def divide_rectangle(x1, y1, x2, y2, n_rect = 9):
    # Calculate width and height of the rectangle
    width = x2 - x1
    height = y2 - y1

    side_length = int(np.sqrt(n_rect))
    # Calculate the size of each sub-rectangle
    sub_width = width / side_length
    sub_height = height / side_length

    # Initialize a list to store the coordinates of the sub-rectangles
    rectangles = []
    
    # Iterate over rows and columns to create sub-rectangles
    for i in range(side_length):
        for j in range(side_length):
            # Calculate coordinates of each sub-rectangle
            sub_x1 = x1 + i * sub_width
            sub_y1 = y1 + j * sub_height
            sub_x2 = sub_x1 + sub_width
            sub_y2 = sub_y1 + sub_height
            
            # Append the coordinates to the list of rectangles
            rectangles.append((sub_x1, sub_y1, sub_x2, sub_y2))
    
    return rectangles


def plot_rectangles(rectangles, values):
    fig, ax = plt.subplots()
    
    # Normalize the values to map them to colors
    norm = Normalize(vmin=min(values), vmax=max(values))
    cmap = plt.get_cmap('coolwarm')  # You can change the colormap here
    
    for rectangle, value in zip(rectangles[::-1], values[::-1]):
        
        x1, y1, x2, y2 = rectangle

        width = x2 - x1
        height = y2 - y1
        color = cmap(norm(value))
        rect = Rectangle((x1, y1), width, height, edgecolor='black', facecolor=color)
        ax.add_patch(rect)
    
    # Add colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Values')
    
    ax.set_aspect('equal', 'box')
    ax.autoscale()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Rectangles')

    plt.xlim((-1.5,1.5))
    plt.ylim((1,4))
    

def plot_strike_zone(df, plot_points=True,balls = None, strikes = None):

    x1 = -.83
    x2 = .83
    y1 = 1.58
    y2 = 3.41
    sub_rectangles = divide_rectangle(x1, y1, x2, y2, n_rect=9)
    sub_rectangles_outside = divide_rectangle(5*x1, y1-2, 5*x2, y2+2, n_rect=4)

    if balls is not None:
        df = df[df.balls==balls]
    if strikes is not None:
        df = df[df.strikes==strikes]
    df['plate_x_batter'] = np.where(df.stand=='L', -df.plate_x, df.plate_x)
    df['plate_z_batter'] = df.plate_z - (df.sz_top - 3.41)
    pdf = df.groupby('zone')\
        .agg({'pred_delta_run_expectancy':'mean',
             'plate_x_batter':'median',
            'plate_z_batter':'median'}).reset_index()
    
    d=pd.DataFrame({'row':list(range(13)), 'zone':[7,4,1,8,5,2,9,6,3,13,11,14,12]})
    pdf = pdf.merge(d, how='left', on = 'zone').sort_values('row')
    pdf = pdf.sort_values('row')
    plot_rectangles(sub_rectangles + sub_rectangles_outside,
                    values = pdf.pred_delta_run_expectancy.values)

    if plot_points:
        sdf = df.sample(500)
        plt.scatter(sdf.plate_x_batter, sdf.plate_z_batter,
                    c=sdf.pred_delta_run_expectancy, cmap =plt.get_cmap('coolwarm'), norm=slope)
    plt.show()
    
def find_sz_edges(df, edge_tolerance = 2/12):
    # give 2 inch edge tolerance

    df['on_edge_left'] = df.plate_x.fillna(0).between(-.83 - edge_tolerance, -.83 + edge_tolerance)
    df['on_edge_right'] = df.plate_x.fillna(0).between(.83 - edge_tolerance, .83 + edge_tolerance)
    
    df['on_edge_top'] = df.plate_z.fillna(0).between(df.sz_top - edge_tolerance, df.sz_top + edge_tolerance)
    df['on_edge_bot'] = df.plate_z.fillna(0).between(df.sz_bot - edge_tolerance, df.sz_bot + edge_tolerance)
    
    df['on_edge_inside'] = np.where(df.stand=='L', df.on_edge_right, df.on_edge_left)
    df['on_edge_outside'] = np.where(df.stand=='R', df.on_edge_right, df.on_edge_left)
    
    df['on_low_inside_corner'] = np.logical_and(df.on_edge_inside, df.on_edge_top)
    df['on_low_outside_corner'] = np.logical_and(df.on_edge_outside, df.on_edge_top)
    
    df['on_high_inside_corner'] = np.logical_and(df.on_edge_inside, df.on_edge_bot)
    df['on_high_outside_corner'] = np.logical_and(df.on_edge_outside, df.on_edge_bot)
    
    df['on_edge'] = df[['on_edge_top','on_edge_bot','on_edge_left','on_edge_right']].sum(axis=1).astype(bool)
    df['on_corner'] = df[['on_high_outside_corner','on_high_inside_corner',
                          'on_low_outside_corner','on_low_inside_corner']].sum(axis=1).astype(bool)
    
    return df





def find_k_pitch_types(df, n_clusters=6, use_pca=True, n_components=4):


    pitch_group_columns = ['velo_percentile_for_pitcher','release_spin_rate',
                           'armside_horz_break','pfx_z', 'armside_tilt', 'spin_efficiency']
    pitch_types = df[pitch_group_columns].dropna()
    
    # Run K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    
    if use_pca:
        # Standardize the features (important for PCA)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(pitch_types)
        
        # Initialize PCA with the number of components you want to retain
        pca = PCA(n_components=n_components)
        
        # Perform PCA
        pca.fit(scaled_data)

        # Transform the data into the new feature space
        pca_data = pca.transform(scaled_data)
        
        # Optionally, you can examine the explained variance ratio
        explained_variance_ratio = pca.explained_variance_ratio_
        print("Explained Variance Ratio:", explained_variance_ratio)

        kmeans.fit(pca_data)
    
    else:
        kmeans.fit(scaled_data)

    # Print cluster labels
    print("Cluster labels:", kmeans.labels_)
    
    # (Optional) Print cluster centers after standardization
    print("Cluster centers (after standardization):")
    print(kmeans.cluster_centers_)

    if use_pca:
        df['est_pitch_type'] = kmeans.predict(
            pca.transform(scaler.transform(df[pitch_group_columns].fillna(0))))
    else:
        df['est_pitch_type'] = kmeans.predict(
            scaler.transform(df[pitch_group_columns].fillna(0)))

    # align the models est pitch type with the most frequent label statcast gives those pitches
    k_pitch_types = df.groupby('est_pitch_type')\
        .agg(k_pitch_type =('pitch_type',pd.Series.mode)).reset_index()
    
    k_pitch_types['count'] = k_pitch_types.\
        groupby('k_pitch_type').est_pitch_type.transform(lambda x:range(1,1+len(x)))
    
    k_pitch_types['total_count'] = k_pitch_types.\
        groupby('k_pitch_type')['count'].transform('max')

    # rename so that if two clusters are FF we get FF_1 and FF_2
    k_pitch_types['k_pitch_type'] = np.where(k_pitch_types.total_count>1, 
                              k_pitch_types.k_pitch_type.astype(str) +'_'+ k_pitch_types['count'].astype(str),
                              k_pitch_types.k_pitch_type.astype(str))

    df = df.merge(k_pitch_types[['k_pitch_type','est_pitch_type']],
         how = 'left', on = 'est_pitch_type').drop(columns = 'est_pitch_type')

    df['k_pitch_type_adj'] = df.groupby(['pitch_type', 'pitcher'])\
        .k_pitch_type.transform(lambda x: x.mode().iloc[0])

    return df



def get_context_stats(df, at_bat_cap = 400):
    
    df['batter_zone_woba'] = 1.2*df\
        .groupby(['batter','zone'])\
        .linear_weight_if_at_bat_over.transform('mean')
    
    df['batter_pitch_woba'] = 1.2*df\
        .groupby(['k_pitch_type_adj'])\
        .linear_weight_if_at_bat_over.transform('mean')


    
    df['batter_count_woba'] = 1.2*df\
        .groupby(['batter','balls', 'strikes']) \
        .linear_weight_if_at_bat_over.transform('mean')
    
    df['batter_zone_sample_size'] = df\
        .groupby(['batter','zone']).end_of_at_bat.transform('sum')
    
    df['batter_pitch_sample_size'] = df\
        .groupby(['batter','k_pitch_type_adj']).end_of_at_bat.transform('sum')
    
    df['batter_count_sample_size'] = df\
        .groupby(['batter','balls', 'strikes']).end_of_at_bat.transform('sum')

    df['batter_sample_size'] = df\
        .groupby(['batter']).end_of_at_bat.transform('sum')
    

    df['count_woba'] = 1.2*df\
    .groupby(['balls', 'strikes']) \
    .linear_weight_if_at_bat_over.transform('mean')
    
    df['zone_woba'] = 1.2*df\
        .groupby(['zone']) \
        .linear_weight_if_at_bat_over.transform('mean')

    df['pitch_woba'] = 1.2*df\
        .groupby(['k_pitch_type_adj']) \
        .linear_weight_if_at_bat_over.transform('mean')

    df['batter_woba'] = 1.2*df\
        .groupby(['batter']) \
        .linear_weight_if_at_bat_over.transform('mean')
    
    df['batter_count_sample_portion'] = (np.sqrt(df['batter_count_sample_size'])/np.sqrt(at_bat_cap)).clip(0,1)
    df['batter_count_woba'] = df['batter_count_sample_portion']*df['batter_count_woba'] + \
        (1-df['batter_count_sample_portion'])*df.count_woba

    df['batter_sample_portion'] = (np.sqrt(df['batter_sample_size'])/np.sqrt(at_bat_cap)).clip(0,1)
    df['batter_woba'] = df['batter_sample_portion']*df['batter_woba'] + \
        (1-df['batter_sample_portion'])*.310
    
    df['batter_zone_sample_portion'] = (np.sqrt(df['batter_zone_sample_size'])/np.sqrt(at_bat_cap)).clip(0,1)
    df['batter_zone_woba'] = df['batter_zone_sample_portion']*df['batter_zone_woba'] + \
        (1-df['batter_zone_sample_portion'])*df.zone_woba
    
    df['batter_pitch_sample_portion'] = (np.sqrt(df['batter_pitch_sample_size'])/np.sqrt(at_bat_cap)).clip(0,1)
    df['batter_pitch_woba'] = df['batter_pitch_sample_portion']*df['batter_pitch_woba'] + \
        (1-df['batter_pitch_sample_portion'])*df.pitch_woba


    # Assuming df is your DataFrame
    pivot_df = df.pivot_table(index='batter', columns='zone', 
                              values='batter_zone_woba', aggfunc='first').reset_index()
    
    # Rename columns
    pivot_df.columns.name = None  # Remove the name of the columns index
    pivot_df.columns = ['batter'] + [f'batter_zone_{col}_woba' for col in pivot_df.columns[1:]]
    
    # Display the pivoted DataFrame
    pivot_df.head()
    
    df = df.merge(pivot_df, how = 'left', on = 'batter')

    pivot_df = df.pivot_table(index='batter', columns='k_pitch_type_adj', 
                              values='batter_pitch_woba', aggfunc='first').reset_index()

    # Rename columns
    pivot_df.columns.name = None  # Remove the name of the columns index
    pivot_df.columns = ['batter'] + [f'batter_pitch_{col}_woba' for col in pivot_df.columns[1:]]
    
    # Display the pivoted DataFrame
    pivot_df.head()
    
    df = df.merge(pivot_df, how = 'left', on = 'batter')
    df = df.drop(columns = [x for x in df.columns if '_sample_' in x])
    df['count_value_away_from_average'] = df.count_value - df.count_value.mean()
    return df

def engineer_features(df):
        
    df = pitch_physics(df)
    df = find_sz_edges(df, edge_tolerance = 2/12)

    df = get_events(df)
    df = get_linear_weights(df)

    df = pitcher_specific_metrics(df)
    

    df['taken'] = np.logical_or(df.description == 'called_strike', df.ball)
    
    df = find_k_pitch_types(df, n_clusters=6, use_pca=True, n_components=4)
    df = get_context_stats(df)
    
    df['delta_run_expectancy'] = np.where(
        df.end_of_at_bat,
        df.linear_weight - df.count_value_away_from_average,
        df.linear_weight)
    
    df.stand = np.where(df.stand=='L',1,0)
    df.p_throws = np.where(df.p_throws=='L',1,0)
    return df

if __name__=='__main__':


    df = pd.read_parquet('pitches_by_season')
    df = engineer_features(df)
    df.to_parquet('engineered_features.parquet')
    
    
 
