


### Utility

def silence_method_call(callback=None, cargs=()):
    with open(os.devnull, 'w') as devnull_file:
        with contextlib.redirect_stdout(devnull_file):
            if callback is not None:
                return callback(*cargs)

def get_dataframe_with_preprocessed_imgs(nb_rows = 1000, img_size = (32, 128), load_pickle_if_exists = True, debug=True, pickle_name="letter_detection_data", with_edge_detection=True):
    full_df = pd.read_pickle('../pickle/df.pickle')
    if not pickle_name:
        raise Exception("Cannot have an empty pickle name")
    pickle_path = "../pickle/" + pickle_name + ".pickle"

    file_exists = os.path.exists(pickle_path)
    if file_exists and load_pickle_if_exists:
        if debug: 
            print("Loading existing data from ", pickle_path, "...")
        return pickle.load(open(pickle_path, "rb"))

    if debug: 
        print("Generating data...")
        

     # Only interested in letters, not punctation or decimal for the moment
    if debug: 
        print("Filtering data: taking only letters")
    r = r'[a-zA-Z]+'
    df = full_df[full_df['transcription'].str.contains(r)]
    np.random.seed(seed=42)

    # reducing row
    if nb_rows >= len(df):
        nb_rows = len(df)
        print('DataFrame only contains', len(df), ' rows => using full dataframe')
    if debug: 
        print("Using", nb_rows, "rows")

    df = df.iloc[random.sample(range(nb_rows), nb_rows)]

    df['length'] = df['transcription'].apply(lambda x: len(x.strip()))
    df.rename(columns = {'form_img_path_y': 'form_img_path'}, inplace = True)
    # reducing columns
    df = df[['michelson_contrast', 'gray_level_mot', 'word_id', 'gray_level', 'x', 'y', 'w', 'h', 'transcription', 'word_img_path', 'form_img_path', 'length']]
    df.reset_index(inplace=True)

    #filtrer les transcriptions vides
    df = df[df['length'] > 0]
    
    if debug: 
        print("Starting preprocessing of images with tensorflow")
        
    try:
        preprocessed_imgs = process_df_img(df, img_size, with_edge_detection=with_edge_detection)
    except:
        print("Unexpected error:", sys.exc_info()[0])
        
    data = {
        'df': df,
        'preprocessed_imgs': preprocessed_imgs
    }
    if debug: 
        print("Creating pickle dump", pickle_path)
    pickle.dump(data, open(pickle_path, "wb" ))
    return data