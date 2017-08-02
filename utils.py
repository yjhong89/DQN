import numpy as np
import os, time

LOG_DIR = './logs'
TRAIN = 'train.csv'
EVAL = 'eval.csv'

def initialize_log():
	try:
		train_log = open(os.path.join(LOG_DIR, TRAIN), 'a')
	except:
		train_log = open(os.path.join(LOG_DIR, TRAIN), 'w')
		train_log.write('Step\t'+',avg_reward\t'+',avg_q\t'+',epsilon\t'+',time\n')
	try:
		eval_log = open(os.path.join(LOG_DIR, EVAL), 'a')
	except:
		eval_log = open(os.path.join(LOG_DIR, EVAL), 'w')
		eval_log.write('Step\t'+',avg_reward\t+'+',avg_q\t'+',epsilon\t'+',time\n')

	return train_log, eval_log


def write_log(steps, total_rwd, total_q, total_loss, num_episode, epsilon, start_time, mode):
	print('At Training step %d, %d episodes => Avg.Q : %3.4f, Avg.rwd : %3.4f, Avg_loss : % 3.4f' % \
		(steps, num_episode, total_q / num_episode, total_rwd / num_episode, total_loss / steps))
	
	train_log, eval_log = initialize_log(mode=mode)
	if mode == 'train':
		train_log.write(str(steps)+'\t,' + str(total_rwd/num_episode)+'\t,' + str(total_q/num_episode)+'\t,' \
		+ str(epsilon) + '\t,' + str(time.time() - start_time) + '\n')
		train_log.flush()
	elif mode == 'eval':
		eval_logs.write(str(steps)+'\t,' + str(total_rwd/num_episode)+'\t,' + str(total_q/num_episode)+'\t,' \
		+ str(epsilon) + '\t,' + str(time.time() - start_time) + '\n')
		eval_log.flush()







	
	
