{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "from multiprocessing import Process, Pool\n",
    "import os, time\n",
    " \n",
    " \n",
    "def main_map(i):\n",
    "    result = i * i\n",
    "    return result\n",
    " \n",
    " \n",
    "if __name__ == '__main__':\n",
    "    inputs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]\n",
    " \n",
    "      # 設定處理程序數量\n",
    "    pool = Pool(4)\n",
    " \n",
    "      # 運行多處理程序\n",
    "    pool_outputs = pool.map(main_map, inputs)\n",
    " \n",
    "      # 輸出執行結果\n",
    "    print(pool_outputs)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  },
  "orig_nbformat": 4,
  "vscode": {
   "interpreter": {
    "hash": "d02331cd1603ec76c6a0d691d59c880477777fc72f4efeafcd9cb0c05515e5a1"
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
