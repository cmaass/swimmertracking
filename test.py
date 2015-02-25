import sys
sys.path.append('/windows/D/goettingen/python/swimmertracking')
import readtraces as rt
mov=rt.movie('/windows/D/datagoe/140507/12.5wtpcTTAB_cut.mp4')
mov.loadTrajectories()
mov.plotMovie('/home/corinna/test.avi', lenlim=50,frate=5)
