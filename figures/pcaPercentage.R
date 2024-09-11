library(ggplot2)
library(gridExtra)
library(cowplot)
library(scales)
library(cowplot)
library(grid)
# install.packages("cowplot")

load("pcaPercentage.RData")
percentage$percentage <- percentage$percentage*100

innovation <- percentage[c(1:6),]
fomc <- percentage[c(7:12),]
complexity <- percentage[c(13:18),]

label_size <- 16
title_size <- 18
bar_width <- 0.7
color_code <- c("#3a9295", "#3a9295", "#3a9295")
border_color <- c("white", "white", "white")
border_width <- 1.1
annotation_size <- 6

p_innovation <- ggplot(data=innovation, aes(x=component, y=percentage)) +
  geom_bar(width=bar_width, stat="identity", color=border_color[1], fill=color_code[1], size=border_width, position=position_dodge()) + 
  theme_bw() +
  theme(legend.position=c(.8,.9), 
        axis.text = element_text(size=label_size),
        axis.title = element_text(size=title_size, face = "italic"),
        legend.title = element_text(size=title_size),
        legend.text = element_text(size=title_size)) +
  geom_text(aes(label=sprintf("%0.2f", percentage)), vjust=-1, color="black",
            position = position_dodge(0.9), size=annotation_size)+
  xlab("") + 
  ylab("") +
  guides(fill=guide_legend(title="")) +
  coord_cartesian(ylim=c(0, 80)) +
  ggtitle("B. Corporate innovation") +
  theme(plot.title = element_text(size=21))


p_fomc <- ggplot(data=fomc, aes(x=component, y=percentage)) +
  geom_bar(width=bar_width, stat="identity", color=border_color[2], fill=color_code[2], size=border_width, position=position_dodge()) + 
  theme_bw() +
  theme(legend.position=c(.8,.9), 
        axis.text = element_text(size=label_size),
        axis.title = element_text(size=title_size, face = "italic"),
        legend.title = element_text(size=title_size),
        legend.text = element_text(size=title_size)) +
  geom_text(aes(label=sprintf("%0.2f", percentage)), vjust=-1, color="black",
            position = position_dodge(0.9), size=annotation_size)+
  xlab("") + 
  ylab("") +
  guides(fill=guide_legend(title="")) +
  coord_cartesian(ylim=c(0, 80)) +
  ggtitle("A. Fed's monetary policy stance") +
  theme(plot.title = element_text(size=21))

p_complexity <- ggplot(data=complexity, aes(x=component, y=percentage)) +
  geom_bar(width=bar_width, stat="identity", color=border_color[3], fill=color_code[3], size=border_width, position=position_dodge()) + 
  theme_bw() +
  theme(legend.position=c(.8,.9), 
        axis.text = element_text(size=label_size),
        axis.title = element_text(size=title_size, face = "italic"),
        legend.title = element_text(size=title_size),
        legend.text = element_text(size=title_size)) +
  geom_text(aes(label=sprintf("%0.2f", percentage)), vjust=-1, color="black",
            position = position_dodge(0.9), size=annotation_size)+
  xlab("") + 
  ylab("") +
  guides(fill=guide_legend(title="")) +
  coord_cartesian(ylim=c(0, 80)) +
  ggtitle("C. Information overload in reviews") +
  theme(plot.title = element_text(size=21))

g <- plot_grid(p_fomc, p_innovation, p_complexity, nrow = 1, ncol = 3)
y.grob <- textGrob("Percentage of explained variances (%)", 
                   gp=gpar(fontface="bold", col="black", fontsize=23), rot=90)
x.grob <- textGrob("Principal components", 
                   gp=gpar(fontface="bold", col="black", fontsize=23))
g <- grid.arrange(arrangeGrob(g, left = y.grob, bottom = x.grob))
ggsave("pcaPercentage.png", g, height = 6, width = 16, dpi = 900)

