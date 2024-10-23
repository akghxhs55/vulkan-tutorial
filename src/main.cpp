#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>

#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <optional>
#include <set>
#include <fstream>
#include <array>

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
    "VK_KHR_portability_subset",
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

constexpr int MAX_FRAMES_IN_FLIGHT = 2;

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif


VkResult createDebugUtilsMessengerEXT(
    const VkInstance instance,
    const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDebugUtilsMessengerEXT* pDebugMessenger)
{
    const auto func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
    if (func == nullptr)
    {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }

    return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
}

void destroyDebugUtilsMessengerEXT(
    const VkInstance instance,
    const VkDebugUtilsMessengerEXT debugMessenger,
    const VkAllocationCallbacks* pAllocator)
{
    const auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
    if (func != nullptr)
    {
        func(instance, debugMessenger, pAllocator);
    }
}

bool checkValidationLayerSupport()
{
    uint32_t layerCount = 0;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char* layerName : validationLayers)
    {
        bool layerFound = false;

        for (const auto& layerProperties : availableLayers)
        {
            if (strcmp(layerName, layerProperties.layerName) == 0)
            {
                layerFound = true;
                break;
            }
        }

        if (!layerFound)
        {
            return false;
        }
    }

    return true;
}

std::vector<const char*> getRequiredExtensionNames()
{
    std::vector<const char*> extensionNames;

    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    for (uint32_t i = 0; i < glfwExtensionCount; ++i)
    {
        extensionNames.emplace_back(glfwExtensions[i]);
    }

    extensionNames.emplace_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);

    if (enableValidationLayers)
    {
        extensionNames.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensionNames;
}

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData)
{
    std::cerr << "Validation layer: " << pCallbackData->pMessage << std::endl;

    return VK_FALSE;
}

void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT &createInfo)
{
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.pNext = nullptr;
    createInfo.flags = 0;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;
    createInfo.pUserData = nullptr;
}

struct QueueFamilyIndices
{
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() const
    {
        return graphicsFamily.has_value() and
                presentFamily.has_value();
    }
};

QueueFamilyIndices findQueueFamilies(const VkPhysicalDevice device, const VkSurfaceKHR surface)
{
    QueueFamilyIndices indices;

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    for (uint32_t i = 0; i < queueFamilyCount; ++i)
    {
        if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
        {
            indices.graphicsFamily.emplace(i);
        }

        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

        if (presentSupport)
        {
            indices.presentFamily.emplace(i);
        }
    }

    return indices;
}

bool checkDeviceExtensionSupport(const VkPhysicalDevice device)
{
    uint32_t availableExtensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &availableExtensionCount, nullptr);

    std::vector<VkExtensionProperties> availableExtensions(availableExtensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &availableExtensionCount, availableExtensions.data());

    std::set<std::string> requiredExtensions(
        deviceExtensions.begin(), deviceExtensions.end());

    for (const auto& extension : availableExtensions)
    {
        requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
}

struct SwapChainSupportDetails
{
    VkSurfaceCapabilitiesKHR capabilities{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

SwapChainSupportDetails querySwapChainSupport(const VkPhysicalDevice device, const VkSurfaceKHR surface)
{
    SwapChainSupportDetails details;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
    if (formatCount != 0)
    {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
    }

    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
    if (presentModeCount != 0)
    {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
    }

    return details;
}

bool isDeviceSuitable(const VkPhysicalDevice device, const VkSurfaceKHR surface)
{
    const QueueFamilyIndices indices = findQueueFamilies(device, surface);

    const bool extensionsSupported = checkDeviceExtensionSupport(device);

    bool swapchainAdequate = false;
    if (extensionsSupported)
    {
        const SwapChainSupportDetails swapchainDetails = querySwapChainSupport(device, surface);
        swapchainAdequate = !swapchainDetails.formats.empty() and
                            !swapchainDetails.presentModes.empty();
    }

    return indices.isComplete() and
            extensionsSupported and
            swapchainAdequate;
}

VkSurfaceFormatKHR getSwapChainSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
{
    for (const auto& availableFormat : availableFormats)
    {
        if (availableFormat.format == VK_FORMAT_B8G8R8_SRGB and
            availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            return availableFormat;
    }

    return availableFormats[0];
}

VkPresentModeKHR getSwapChainPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes)
{
    for (const auto& availablePresentMode : availablePresentModes)
    {
        if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
        {
            return availablePresentMode;
        }
    }

    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D getSwapChainExtent(const VkSurfaceCapabilitiesKHR& capabilities, GLFWwindow* window)
{
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
    {
        return capabilities.currentExtent;
    }

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    VkExtent2D actualExtent = {
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height)
    };

    actualExtent.width = std::clamp(
        actualExtent.width,
        capabilities.minImageExtent.width,
        capabilities.maxImageExtent.width);
    actualExtent.height = std::clamp(
        actualExtent.height,
        capabilities.minImageExtent.height,
        capabilities.maxImageExtent.height);

    return actualExtent;
}

std::vector<char> readFile(const std::string& filename)
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    const size_t fileSize = static_cast<size_t>(file.tellg());

    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), static_cast<std::streamsize>(fileSize));

    file.close();

    return buffer;
}

VkShaderModule getShaderModule(const std::vector<char>& code, const VkDevice device)
{
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.pNext = nullptr;
    createInfo.flags = 0;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    if (const VkResult result = vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule); result != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create shader module with error code: " + std::to_string(result));
    }

    return shaderModule;
}

VkViewport getViewport(const VkExtent2D extent)
{
    VkViewport viewport{};

    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(extent.width);
    viewport.height = static_cast<float>(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    return viewport;
}

VkRect2D getScissor(const VkExtent2D extent)
{
    VkRect2D scissor{};

    scissor.offset = {0, 0};
    scissor.extent = extent;

    return scissor;
}

struct Vertex
{
    glm::vec2 pos;
    glm::vec3 color;

    static VkVertexInputBindingDescription getBindingDescription()
    {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions()
    {
        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, color);

        return attributeDescriptions;
    }
};

const std::vector<Vertex> vertices = {
    {{0.0f, -0.5f}, {1.0f, 1.0f, 1.0f}},
    {{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
    {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}
};

uint32_t findMemoryType(const uint32_t typeFilter, const VkMemoryPropertyFlags properties, const VkPhysicalDevice physicalDevice)
{
    VkPhysicalDeviceMemoryProperties memoryProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++)
    {
        if ((typeFilter & (1 << i)) and
            (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
        {
            return i;
        }
    }

    throw std::runtime_error("Failed to find suitable memory type.");
}


class HelloTriangleApplication
{
public:
    void run()
    {
        initiateWindow();
        initiateVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* window = nullptr;
    VkInstance instance = nullptr;
    VkDebugUtilsMessengerEXT debugMessenger = nullptr;
    VkSurfaceKHR surface = nullptr;
    VkPhysicalDevice physicalDevice = nullptr;
    VkDevice device = nullptr;
    VkQueue graphicsQueue = nullptr;
    VkQueue presentQueue = nullptr;
    VkSwapchainKHR swapchain = nullptr;
    VkFormat swapchainImageFormat = VK_FORMAT_UNDEFINED;
    VkExtent2D swapchainExtent = {};
    std::vector<VkImage> swapchainImages;
    std::vector<VkImageView> swapchainImageViews;
    VkRenderPass renderPass = nullptr;
    VkPipelineLayout pipelineLayout = nullptr;
    VkPipeline graphicsPipeline = nullptr;
    std::vector<VkFramebuffer> swapchainFramebuffers;
    VkCommandPool commandPool = nullptr;
    VkBuffer vertexBuffer = nullptr;
    VkDeviceMemory vertexBufferMemory = nullptr;
    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    uint32_t currentFrame = 0;
    bool framebufferResized = false;

    void initiateWindow()
    {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    }

    void initiateVulkan()
    {
        createInstance();
        if (enableValidationLayers)
        {
            createDebugMessenger();
        }
        createSurface();
        setupPhysicalDevice();
        createDevice();
        setupQueues();
        createSwapchain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandPool();
        createVertexBuffer();
        bindBuffer();
        createCommandBuffer();
        createSyncObjects();
    }

    void mainLoop()
    {
        while (!glfwWindowShouldClose(window))
        {
            glfwPollEvents();
            drawFrame();
        }

        vkDeviceWaitIdle(device);
    }

    void cleanup() const
    {
        for (const auto fence : inFlightFences)
        {
            vkDestroyFence(device, fence, nullptr);
        }
        for (const auto semaphore : renderFinishedSemaphores)
        {
            vkDestroySemaphore(device, semaphore, nullptr);
        }
        for (const auto semaphore : imageAvailableSemaphores)
        {
            vkDestroySemaphore(device, semaphore, nullptr);
        }
        vkDestroyBuffer(device, vertexBuffer, nullptr);
        vkFreeMemory(device, vertexBufferMemory, nullptr);
        vkDestroyCommandPool(device, commandPool, nullptr);
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyRenderPass(device, renderPass, nullptr);
        cleanupSwapchain();
        vkDestroyDevice(device, nullptr);
        vkDestroySurfaceKHR(instance, surface, nullptr);
        if (enableValidationLayers)
        {
            destroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }
        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);

        glfwTerminate();
    }

    void createInstance()
    {
        if (enableValidationLayers and !checkValidationLayerSupport())
        {
            throw std::runtime_error("Validation layers requested, but not available.");
        }

        VkApplicationInfo applicationInfo{};
        applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        applicationInfo.pNext = nullptr;
        applicationInfo.pApplicationName = "Hello Triangle";
        applicationInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        applicationInfo.pEngineName = "No Engine";
        applicationInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        applicationInfo.apiVersion = VK_API_VERSION_1_3;

        const void* pNext = nullptr;
        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
        if (enableValidationLayers)
        {
            populateDebugMessengerCreateInfo(debugCreateInfo);
            pNext = &debugCreateInfo;
        }

        std::vector<const char*> layerNames;
        if (enableValidationLayers)
        {
            layerNames = validationLayers;
        }

        const std::vector<const char*> extensionNames = getRequiredExtensionNames();

        VkInstanceCreateInfo instanceCreateInfo{};
        instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instanceCreateInfo.pNext = pNext;
        instanceCreateInfo.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        instanceCreateInfo.pApplicationInfo = &applicationInfo;
        instanceCreateInfo.enabledLayerCount = static_cast<uint32_t>(layerNames.size());
        instanceCreateInfo.ppEnabledLayerNames = layerNames.data();
        instanceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(extensionNames.size());
        instanceCreateInfo.ppEnabledExtensionNames = extensionNames.data();

        if(const VkResult result = vkCreateInstance(&instanceCreateInfo, nullptr, &instance); result != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create Vulkan instance with error code: " + std::to_string(result));
        }
    }

    void createDebugMessenger()
    {
        VkDebugUtilsMessengerCreateInfoEXT createInfo{};
        populateDebugMessengerCreateInfo(createInfo);

        if (createDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to set up debug messenger.");
        }
    }

    void createSurface()
    {
        if (const VkResult result = glfwCreateWindowSurface(instance, window, nullptr, &surface); result != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create window surface with error code: " + std::to_string(result));
        }
    }

    void setupPhysicalDevice()
    {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        if (deviceCount == 0)
        {
            throw std::runtime_error("Failed to find GPUs with Vulkan support.");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        for (const auto& device : devices)
        {
            if (isDeviceSuitable(device, surface))
            {
                physicalDevice = device;
                break;
            }
        }

        if (physicalDevice == nullptr)
        {
            throw std::runtime_error("Failed to find a suitable GPU.");
        }
    }

    void createDevice()
    {
        const QueueFamilyIndices indices = findQueueFamilies(physicalDevice, surface);

        std::set<uint32_t> uniqueQueueFamilies = {
            indices.graphicsFamily.value(),
            indices.presentFamily.value()};
        constexpr float queuePriority = 1.0f;

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        for (uint32_t queueFamily : uniqueQueueFamilies)
        {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.pNext = nullptr;
            queueCreateInfo.flags = 0;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;

            queueCreateInfos.emplace_back(queueCreateInfo);
        }

        constexpr VkPhysicalDeviceFeatures features{};

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.pNext = nullptr;
        createInfo.flags = 0;
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();
        createInfo.pEnabledFeatures = &features;

        if (const VkResult result = vkCreateDevice(physicalDevice, &createInfo, nullptr, &device); result != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create logical device with error code: " + std::to_string(result));
        }
    }

    void setupQueues()
    {
        const QueueFamilyIndices indices = findQueueFamilies(physicalDevice, surface);

        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
    }

    void createSwapchain()
    {
        const SwapChainSupportDetails swapchainSupportDetails = querySwapChainSupport(physicalDevice, surface);

        uint32_t minImageCount = swapchainSupportDetails.capabilities.minImageCount + 1;
        if (swapchainSupportDetails.capabilities.maxImageCount > 0 and
            swapchainSupportDetails.capabilities.maxImageCount < minImageCount)
        {
            minImageCount = swapchainSupportDetails.capabilities.maxImageCount;
        }

        const VkSurfaceFormatKHR surfaceFormat = getSwapChainSurfaceFormat(swapchainSupportDetails.formats);

        const VkExtent2D extent = getSwapChainExtent(swapchainSupportDetails.capabilities, window);

        VkSharingMode sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        const QueueFamilyIndices indices = findQueueFamilies(physicalDevice, surface);
        std::vector<uint32_t> queueFamilyIndices;
        if (indices.graphicsFamily.value() != indices.presentFamily.value())
        {
            sharingMode = VK_SHARING_MODE_CONCURRENT;
            queueFamilyIndices = {indices.graphicsFamily.value(), indices.presentFamily.value()};
        }

        const VkPresentModeKHR presentMode = getSwapChainPresentMode(swapchainSupportDetails.presentModes);

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.pNext = nullptr;
        createInfo.flags = 0;
        createInfo.surface = surface;
        createInfo.minImageCount = minImageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        createInfo.imageSharingMode = sharingMode;
        createInfo.queueFamilyIndexCount = static_cast<uint32_t>(queueFamilyIndices.size());
        createInfo.pQueueFamilyIndices = queueFamilyIndices.data();
        createInfo.preTransform = swapchainSupportDetails.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;
        createInfo.oldSwapchain = nullptr;

        if (const VkResult result = vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapchain); result != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create swap chain with error code: " + std::to_string(result));
        }

        swapchainImageFormat = surfaceFormat.format;

        swapchainExtent = extent;

        uint32_t imageCount = 0;
        vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr);
        swapchainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapchain, &imageCount, swapchainImages.data());

    }

    void createImageViews()
    {
        swapchainImageViews.resize(swapchainImages.size());

        for (size_t i = 0; i < swapchainImages.size(); i++)
        {
            VkComponentMapping components{};
            components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

            VkImageSubresourceRange subresourceRange{};
            subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            subresourceRange.baseMipLevel = 0;
            subresourceRange.levelCount = 1;
            subresourceRange.baseArrayLayer = 0;
            subresourceRange.layerCount = 1;

            VkImageViewCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.pNext = nullptr;
            createInfo.flags = 0;
            createInfo.image = swapchainImages[i];
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format = swapchainImageFormat;
            createInfo.components = components;
            createInfo.subresourceRange = subresourceRange;

            if (const VkResult result = vkCreateImageView(device, &createInfo, nullptr, &swapchainImageViews[i]); result != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create image views with error code: " + std::to_string(result));
            }
        }
    }

    void createRenderPass()
    {
        VkAttachmentDescription colorAttachmentDescription{};
        colorAttachmentDescription.flags = 0;
        colorAttachmentDescription.format = swapchainImageFormat;
        colorAttachmentDescription.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachmentDescription.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachmentDescription.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachmentDescription.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachmentDescription.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachmentDescription.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachmentDescription.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference colorAttachmentReference{};
        colorAttachmentReference.attachment = 0;
        colorAttachmentReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpassDescription{};
        subpassDescription.flags = 0;
        subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpassDescription.inputAttachmentCount = 0;
        subpassDescription.pInputAttachments = nullptr;
        subpassDescription.colorAttachmentCount = 1;
        subpassDescription.pColorAttachments = &colorAttachmentReference;
        subpassDescription.pResolveAttachments = nullptr;
        subpassDescription.pDepthStencilAttachment = nullptr;
        subpassDescription.preserveAttachmentCount = 0;
        subpassDescription.pPreserveAttachments = nullptr;

        VkSubpassDependency subpassDependency{};
        subpassDependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        subpassDependency.dstSubpass = 0;
        subpassDependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        subpassDependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        subpassDependency.srcAccessMask = 0;
        subpassDependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        subpassDependency.dependencyFlags = 0;

        VkRenderPassCreateInfo renderPassCreateInfo{};
        renderPassCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassCreateInfo.pNext = nullptr;
        renderPassCreateInfo.flags = 0;
        renderPassCreateInfo.attachmentCount = 1;
        renderPassCreateInfo.pAttachments = &colorAttachmentDescription;
        renderPassCreateInfo.subpassCount = 1;
        renderPassCreateInfo.pSubpasses = &subpassDescription;
        renderPassCreateInfo.dependencyCount = 1;
        renderPassCreateInfo.pDependencies = &subpassDependency;

        if (const VkResult result = vkCreateRenderPass(device, &renderPassCreateInfo, nullptr, &renderPass); result != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create render pass with error code: " + std::to_string(result));
        }
    }

    void createGraphicsPipeline()
    {
        const std::vector<char> vertexShaderCode = readFile("../shaders/vertex.spv");
        const std::vector<char> fragmentShaderCode = readFile("../shaders/fragment.spv");

        const VkShaderModule vertexShaderModule = getShaderModule(vertexShaderCode, device);
        const VkShaderModule fragmentShaderModule = getShaderModule(fragmentShaderCode, device);

        VkPipelineShaderStageCreateInfo vertexShaderStageCreateInfo{};
        vertexShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertexShaderStageCreateInfo.pNext = nullptr;
        vertexShaderStageCreateInfo.flags = 0;
        vertexShaderStageCreateInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertexShaderStageCreateInfo.module = vertexShaderModule;
        vertexShaderStageCreateInfo.pName = "main";
        vertexShaderStageCreateInfo.pSpecializationInfo = nullptr;

        VkPipelineShaderStageCreateInfo fragmentShaderStageCreateInfo{};
        fragmentShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragmentShaderStageCreateInfo.pNext = nullptr;
        fragmentShaderStageCreateInfo.flags = 0;
        fragmentShaderStageCreateInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragmentShaderStageCreateInfo.module = fragmentShaderModule;
        fragmentShaderStageCreateInfo.pName = "main";
        fragmentShaderStageCreateInfo.pSpecializationInfo = nullptr;

        std::vector<VkPipelineShaderStageCreateInfo> shaderStageCreateInfos = {
            vertexShaderStageCreateInfo,
            fragmentShaderStageCreateInfo};

        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        VkPipelineVertexInputStateCreateInfo vertexInputStateCreateInfo{};
        vertexInputStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputStateCreateInfo.pNext = nullptr;
        vertexInputStateCreateInfo.flags = 0;
        vertexInputStateCreateInfo.vertexBindingDescriptionCount = 1;
        vertexInputStateCreateInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputStateCreateInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputStateCreateInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCreateInfo{};
        inputAssemblyStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssemblyStateCreateInfo.pNext = nullptr;
        inputAssemblyStateCreateInfo.flags = 0;
        inputAssemblyStateCreateInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssemblyStateCreateInfo.primitiveRestartEnable = VK_FALSE;

        const VkViewport viewport = getViewport(swapchainExtent);

        const VkRect2D scissor = getScissor(swapchainExtent);

        VkPipelineViewportStateCreateInfo viewportStateCreateInfo{};
        viewportStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportStateCreateInfo.pNext = nullptr;
        viewportStateCreateInfo.flags = 0;
        viewportStateCreateInfo.viewportCount = 1;
        viewportStateCreateInfo.pViewports = &viewport;
        viewportStateCreateInfo.scissorCount = 1;
        viewportStateCreateInfo.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo rasterizationStateCreateInfo{};
        rasterizationStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizationStateCreateInfo.pNext = nullptr;
        rasterizationStateCreateInfo.flags = 0;
        rasterizationStateCreateInfo.depthClampEnable = VK_FALSE;
        rasterizationStateCreateInfo.rasterizerDiscardEnable = VK_FALSE;
        rasterizationStateCreateInfo.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizationStateCreateInfo.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizationStateCreateInfo.frontFace = VK_FRONT_FACE_CLOCKWISE;
        rasterizationStateCreateInfo.depthBiasEnable = VK_FALSE;
        rasterizationStateCreateInfo.depthBiasConstantFactor = 0.0f;
        rasterizationStateCreateInfo.depthBiasClamp = 0.0f;
        rasterizationStateCreateInfo.depthBiasSlopeFactor = 0.0f;
        rasterizationStateCreateInfo.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo multisampleStateCreateInfo{};
        multisampleStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampleStateCreateInfo.pNext = nullptr;
        multisampleStateCreateInfo.flags = 0;
        multisampleStateCreateInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisampleStateCreateInfo.sampleShadingEnable = VK_FALSE;
        multisampleStateCreateInfo.minSampleShading = 1.0f;
        multisampleStateCreateInfo.pSampleMask = nullptr;
        multisampleStateCreateInfo.alphaToCoverageEnable = VK_FALSE;
        multisampleStateCreateInfo.alphaToOneEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.blendEnable = VK_FALSE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                                              VK_COLOR_COMPONENT_G_BIT |
                                              VK_COLOR_COMPONENT_B_BIT |
                                              VK_COLOR_COMPONENT_A_BIT;

        VkPipelineColorBlendStateCreateInfo colorBlendStateCreateInfo{};
        colorBlendStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlendStateCreateInfo.pNext = nullptr;
        colorBlendStateCreateInfo.flags = 0;
        colorBlendStateCreateInfo.logicOpEnable = VK_FALSE;
        colorBlendStateCreateInfo.logicOp = VK_LOGIC_OP_COPY;
        colorBlendStateCreateInfo.attachmentCount = 1;
        colorBlendStateCreateInfo.pAttachments = &colorBlendAttachment;
        colorBlendStateCreateInfo.blendConstants[0] = 0.0f;
        colorBlendStateCreateInfo.blendConstants[1] = 0.0f;
        colorBlendStateCreateInfo.blendConstants[2] = 0.0f;
        colorBlendStateCreateInfo.blendConstants[3] = 0.0f;

        const std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };

        VkPipelineDynamicStateCreateInfo dynamicStateCreateInfo{};
        dynamicStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicStateCreateInfo.pNext = nullptr;
        dynamicStateCreateInfo.flags = 0;
        dynamicStateCreateInfo.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicStateCreateInfo.pDynamicStates = dynamicStates.data();

        VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
        pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutCreateInfo.pNext = nullptr;
        pipelineLayoutCreateInfo.flags = 0;
        pipelineLayoutCreateInfo.setLayoutCount = 0;
        pipelineLayoutCreateInfo.pSetLayouts = nullptr;
        pipelineLayoutCreateInfo.pushConstantRangeCount = 0;
        pipelineLayoutCreateInfo.pPushConstantRanges = nullptr;

        if (const VkResult result = vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout); result != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create pipeline layout with error code: " + std::to_string(result));
        }

        VkGraphicsPipelineCreateInfo pipelineCreateInfo{};
        pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineCreateInfo.pNext = nullptr;
        pipelineCreateInfo.flags = 0;
        pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStageCreateInfos.size());
        pipelineCreateInfo.pStages = shaderStageCreateInfos.data();
        pipelineCreateInfo.pVertexInputState = &vertexInputStateCreateInfo;
        pipelineCreateInfo.pInputAssemblyState = &inputAssemblyStateCreateInfo;
        pipelineCreateInfo.pTessellationState = nullptr;
        pipelineCreateInfo.pViewportState = &viewportStateCreateInfo;
        pipelineCreateInfo.pRasterizationState = &rasterizationStateCreateInfo;
        pipelineCreateInfo.pMultisampleState = &multisampleStateCreateInfo;
        pipelineCreateInfo.pDepthStencilState = nullptr;
        pipelineCreateInfo.pColorBlendState = &colorBlendStateCreateInfo;
        pipelineCreateInfo.pDynamicState = &dynamicStateCreateInfo;
        pipelineCreateInfo.layout = pipelineLayout;
        pipelineCreateInfo.renderPass = renderPass;
        pipelineCreateInfo.subpass = 0;
        pipelineCreateInfo.basePipelineHandle = nullptr;
        pipelineCreateInfo.basePipelineIndex = -1;

        if (const VkResult result = vkCreateGraphicsPipelines(device, nullptr, 1, &pipelineCreateInfo, nullptr, &graphicsPipeline); result != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create graphics pipeline with error code: " + std::to_string(result));
        }

        vkDestroyShaderModule(device, fragmentShaderModule, nullptr);
        vkDestroyShaderModule(device, vertexShaderModule, nullptr);
    }

    void createFramebuffers()
    {
        swapchainFramebuffers.resize(swapchainImageViews.size());

        for (size_t i = 0; i < swapchainImageViews.size(); i++)
        {
            std::vector<VkImageView> attachments = {
                swapchainImageViews[i]
            };

            VkFramebufferCreateInfo framebufferCreateInfo{};
            framebufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferCreateInfo.pNext = nullptr;
            framebufferCreateInfo.flags = 0;
            framebufferCreateInfo.renderPass = renderPass;
            framebufferCreateInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            framebufferCreateInfo.pAttachments = attachments.data();
            framebufferCreateInfo.width = swapchainExtent.width;
            framebufferCreateInfo.height = swapchainExtent.height;
            framebufferCreateInfo.layers = 1;

            if (const VkResult result = vkCreateFramebuffer(device, &framebufferCreateInfo, nullptr, &swapchainFramebuffers[i]); result != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create framebuffer with error code: " + std::to_string(result));
            }
        }
    }

    void createVertexBuffer()
    {
        VkBufferCreateInfo bufferCreateInfo{};
        bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferCreateInfo.pNext = nullptr;
        bufferCreateInfo.flags = 0;
        bufferCreateInfo.size = sizeof(vertices[0]) * vertices.size();
        bufferCreateInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        bufferCreateInfo.queueFamilyIndexCount = 0;
        bufferCreateInfo.pQueueFamilyIndices = nullptr;

        if (const VkResult result = vkCreateBuffer(device, &bufferCreateInfo, nullptr, &vertexBuffer); result != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create vertex buffer with error code: " + std::to_string(result));
        }
    }

    void bindBuffer()
    {
        VkMemoryRequirements memoryRequirements;
        vkGetBufferMemoryRequirements(device, vertexBuffer, &memoryRequirements);

        VkMemoryAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocateInfo.pNext = nullptr;
        allocateInfo.allocationSize = memoryRequirements.size;
        allocateInfo.memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, physicalDevice);

        if (const VkResult result = vkAllocateMemory(device, &allocateInfo, nullptr, &vertexBufferMemory); result != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to allocate vertex buffer memory with error code: " + std::to_string(result));
        }

        vkBindBufferMemory(device, vertexBuffer, vertexBufferMemory, 0);

        void *data;
        vkMapMemory(device, vertexBufferMemory, 0, sizeof(vertices[0]) * vertices.size(), 0, &data);
        memcpy(data, vertices.data(), sizeof(vertices[0]) * vertices.size());

        vkUnmapMemory(device, vertexBufferMemory);
    }

    void createCommandPool()
    {
        const QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice, surface);

        VkCommandPoolCreateInfo commandPoolCreateInfo{};
        commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        commandPoolCreateInfo.pNext = nullptr;
        commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        commandPoolCreateInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

        if (const VkResult result = vkCreateCommandPool(device, &commandPoolCreateInfo, nullptr, &commandPool); result != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create command pool with error code: " + std::to_string(result));
        }
    }

    void createCommandBuffer()
    {
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

        VkCommandBufferAllocateInfo commandBufferAllocateInfo{};
        commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandBufferAllocateInfo.pNext = nullptr;
        commandBufferAllocateInfo.commandPool = commandPool;
        commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandBufferAllocateInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());

        if (const VkResult result = vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, commandBuffers.data()); result != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to allocate command buffer with error code: " + std::to_string(result));
        }
    }

    void createSyncObjects()
    {
        VkSemaphoreCreateInfo semaphoreCreateInfo{};
        semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        semaphoreCreateInfo.pNext = nullptr;
        semaphoreCreateInfo.flags = 0;

        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        for (auto& semaphore : imageAvailableSemaphores)
        {
            if (const VkResult result = vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &semaphore); result != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create image available semaphore with error code: " + std::to_string(result));
            }
        }

        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        for (auto& semaphore : renderFinishedSemaphores)
        {
            if (const VkResult result = vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &semaphore); result != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create render finished semaphore with error code: " + std::to_string(result));
            }
        }

        VkFenceCreateInfo fenceCreateInfo{};
        fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceCreateInfo.pNext = nullptr;
        fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
        for (auto& fence : inFlightFences)
        {
            if (const VkResult result = vkCreateFence(device, &fenceCreateInfo, nullptr, &fence); result != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create in flight fence with error code: " + std::to_string(result));
            }
        }
    }

    void drawFrame()
    {
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, std::numeric_limits<uint64_t>::max());

        uint32_t imageIndex;
        const VkResult acquireResult = vkAcquireNextImageKHR(device, swapchain, std::numeric_limits<uint64_t>::max(), imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
        if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR)
        {
            recreateSwapchain();
            return;
        }
        else if (acquireResult != VK_SUCCESS and
                acquireResult != VK_SUBOPTIMAL_KHR)
        {
            throw std::runtime_error("Failed to acquire next image with error code: " + std::to_string(acquireResult));
        }

        vkResetFences(device, 1, &inFlightFences[currentFrame]);

        vkResetCommandBuffer(commandBuffers[currentFrame], 0);
        recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

        const std::vector<VkSemaphore> waitSemaphores = {imageAvailableSemaphores[currentFrame]};
        const std::vector<VkPipelineStageFlags> waitStages = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        const std::vector<VkSemaphore> signalSemaphores = {renderFinishedSemaphores[currentFrame]};

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.pNext = nullptr;
        submitInfo.waitSemaphoreCount = static_cast<uint32_t>(waitSemaphores.size());
        submitInfo.pWaitSemaphores = waitSemaphores.data();
        submitInfo.pWaitDstStageMask = waitStages.data();
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentFrame];
        submitInfo.signalSemaphoreCount = static_cast<uint32_t>(signalSemaphores.size());
        submitInfo.pSignalSemaphores = signalSemaphores.data();

        if (const VkResult result = vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]); result != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to submit draw command buffer with error code: " + std::to_string(result));
        }

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.pNext = nullptr;
        presentInfo.waitSemaphoreCount = static_cast<uint32_t>(signalSemaphores.size());
        presentInfo.pWaitSemaphores = signalSemaphores.data();
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapchain;
        presentInfo.pImageIndices = &imageIndex;
        presentInfo.pResults = nullptr;

        const VkResult presentResult = vkQueuePresentKHR(presentQueue, &presentInfo);
        if (presentResult == VK_ERROR_OUT_OF_DATE_KHR or
            presentResult == VK_SUBOPTIMAL_KHR or
            framebufferResized)
        {
            framebufferResized = false;
            recreateSwapchain();
        }
        else if (presentResult != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to present image with error code: " + std::to_string(presentResult));
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void recordCommandBuffer(const VkCommandBuffer commandBuffer, const uint32_t imageIndex) const
    {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.pNext = nullptr;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
        beginInfo.pInheritanceInfo = nullptr;

        if (const VkResult result = vkBeginCommandBuffer(commandBuffer, &beginInfo); result != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to begin recording command buffer with error code: " + std::to_string(result));
        }

        constexpr VkClearValue clearColor = {0.0f, 0.0f, 0.0f, 1.0f};

        VkRenderPassBeginInfo renderPassBeginInfo{};
        renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassBeginInfo.pNext = nullptr;
        renderPassBeginInfo.renderPass = renderPass;
        renderPassBeginInfo.framebuffer = swapchainFramebuffers[imageIndex];
        renderPassBeginInfo.renderArea.offset = {0, 0};
        renderPassBeginInfo.renderArea.extent = swapchainExtent;
        renderPassBeginInfo.clearValueCount = 1;
        renderPassBeginInfo.pClearValues = &clearColor;

        vkCmdBeginRenderPass(commandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

        const VkViewport viewport = getViewport(swapchainExtent);
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        const VkRect2D scissor = getScissor(swapchainExtent);
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        const std::vector<VkBuffer> vertexBuffers = {vertexBuffer};
        constexpr VkDeviceSize offsets[] = {0};
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers.data(), offsets);

        vkCmdDraw(commandBuffer, static_cast<uint32_t>(vertices.size()), 1, 0, 0);

        vkCmdEndRenderPass(commandBuffer);

        if (const VkResult result = vkEndCommandBuffer(commandBuffer); result != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to record command buffer with error code: " + std::to_string(result));
        }
    }

    void recreateSwapchain()
    {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 or height == 0)
        {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(device);

        createSwapchain();
        createImageViews();
        createFramebuffers();
    }

    void cleanupSwapchain() const
    {
        for (const auto framebuffer : swapchainFramebuffers)
        {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }
        for (const auto imageView : swapchainImageViews)
        {
            vkDestroyImageView(device, imageView, nullptr);
        }
        vkDestroySwapchainKHR(device, swapchain, nullptr);
    }

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height)
    {
        const auto app = static_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }
};


int main()
{
    try
    {
        HelloTriangleApplication app;
        app.run();
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
