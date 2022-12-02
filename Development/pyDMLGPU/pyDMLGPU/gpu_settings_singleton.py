import pyopencl as cl


class GPUSetter:
    __instance = None

    class __Holder:
        def __init__(self):
            self.platform = cl.get_platforms()[0]
            self.device = self.platform.get_devices(cl.device_type.GPU)[0]
            self.context = cl.Context([self.device])

        def update_device(self):
            self.device = self.platform.get_devices(cl.device_type.GPU)[0]

        def update_context(self):
            self.context = cl.Context([self.device])

    def __init__(self):
        if GPUSetter.__instance is None:
            GPUSetter.__instance = GPUSetter.__Holder()

    @staticmethod
    def get_platforms():
        return cl.get_platforms()

    def set_platform(self, index):
        try:
            self.__instance.platform = self.get_platforms()[index]
            self.__instance.update_device()
            self.__instance.update_context()
        except IndexError:
            print("OUT OF LENGTH")

    def get_devices(self):
        return self.__instance.platform.get_devices(cl.device_type.GPU)

    def get_device(self):
        return self.__instance.device

    def set_device(self, index):
        try:
            self.__instance.device = self.get_devices()[index]
            self.__instance.update_context()
        except IndexError:
            print("OUT OF LENGTH")

    def get_context(self):
        return self.__instance.context

    def make_queue(self):
        return cl.CommandQueue(self.get_context())


# class A:
#     def __init__(self):
#         self.s = GPUSetter()
#
#
# class B:
#     def __init__(self):
#         self.s = GPUSetter()
#
#
# if __name__ == '__main__':
#     a = A()
#     b = B()
#     c = GPUSetter()
#     print('platforms', a.s.get_platforms())
#     print('devices', a.s.get_devices())
#     print('context', c.get_context())
#     c.set_platform(1)
#     print('queue', c.make_queue())
