class HyperParameters:
      def __init__(self, 
      n_embd=32, 
      vocab_size=-1,
      block_size=8, 
      batch_size=32, 
      head_size=32, 
      max_iterations=1000, 
      eval_interval=300, 
      learning_rate=1e-3, 
      mps_is_available=False, 
      use_tiktoken=False, 
      training_split=0.9, 
      device='cpu'):
      
         super().__init__()
         self.learning_rate=learning_rate
         self.n_embd=n_embd
         self.vocab_size = vocab_size
         self.block_size = block_size
         self.batch_size = batch_size
         self.head_size = head_size
         self.max_iterations = max_iterations
         self.eval_interval = eval_interval
         self.learning_rate = eval_interval
         self.mps_is_available = mps_is_available
         self.use_tiktoken = use_tiktoken
         self.training_split = training_split
         self.device = device