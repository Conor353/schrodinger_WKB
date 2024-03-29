#Plot relative error


# setting the axes at the centre
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)


ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

plt.plot(x,rel_error,'red', label='Error')
plt.legend(loc='upper left')

# show the plot
plt.show()